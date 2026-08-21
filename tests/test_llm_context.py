"""Context capacity estimation, and the calibration that replaces its guess.

Section 12's requirement is that the estimate be model-specific rather than a
tokens-per-gigabyte constant, so the tests that matter here are the ones that
would still pass against a constant and must not: two models of the same size
with different attention shapes have to produce different answers.
"""

from __future__ import annotations

import pytest

import mc_gguf
import mc_llm_context as ctx
from test_gguf import _text, _u32, _u32s, write_gguf  # noqa: F401

_GB = 1024**3
_MB = 1024**2


def build_model(tmp_path, name="model.gguf", *, blocks=32, heads=32, kv_heads=8,
                embedding=4096, context=8192, size_mb=4):
    metadata = b"".join([
        _text("general.architecture", "llama"),
        _u32("llama.block_count", blocks),
        _u32("llama.context_length", context),
        _u32("llama.embedding_length", embedding),
        _u32("llama.attention.head_count", heads),
        _u32("llama.attention.head_count_kv", kv_heads),
    ])
    return write_gguf(tmp_path / name, metadata, 6, padding=size_mb * _MB)


@pytest.fixture(autouse=True)
def calibration(tmp_path, monkeypatch):
    """Point calibration at a throwaway file and empty it between tests."""
    monkeypatch.setattr(ctx, "_store_path", lambda: tmp_path / "calibration.json")
    ctx.forget()
    yield
    ctx.forget()


class TestCacheArithmetic:
    def test_the_cost_per_token_follows_the_model_shape(self, tmp_path):
        model = mc_gguf.read(build_model(tmp_path))
        placement = ctx.Placement(context=8192)

        # 32 blocks x 8 KV heads x (128 + 128) x 2 bytes for f16.
        assert ctx.kv_bytes_per_token(model, placement) == 32 * 8 * (128 + 128) * 2.0

    def test_grouped_query_attention_costs_a_quarter_of_multi_head(self, tmp_path):
        """The case a tokens-per-gigabyte constant gets wrong. Same width, same
        depth, four times fewer KV heads, four times smaller cache."""
        grouped = mc_gguf.read(build_model(tmp_path, "gqa.gguf", kv_heads=8))
        multi = mc_gguf.read(build_model(tmp_path, "mha.gguf", kv_heads=32))
        placement = ctx.Placement()

        assert (ctx.kv_bytes_per_token(multi, placement)
                == 4 * ctx.kv_bytes_per_token(grouped, placement))

    def test_quantising_the_cache_shrinks_it_by_the_element_size(self, tmp_path):
        model = mc_gguf.read(build_model(tmp_path))
        full = ctx.kv_bytes_per_token(model, ctx.Placement())
        quantised = ctx.kv_bytes_per_token(
            model, ctx.Placement(kv_type_k="q8_0", kv_type_v="q8_0"))

        assert quantised == pytest.approx(full * (34 / 32) / 2.0)

    def test_the_two_halves_are_sized_separately(self, tmp_path):
        """llama.cpp quantises K and V independently, so a mixed setting has to
        land between the two uniform ones rather than at one of them."""
        model = mc_gguf.read(build_model(tmp_path))
        mixed = ctx.kv_bytes_per_token(model, ctx.Placement(kv_type_k="q8_0"))
        full = ctx.kv_bytes_per_token(model, ctx.Placement())
        both = ctx.kv_bytes_per_token(model, ctx.Placement(kv_type_k="q8_0", kv_type_v="q8_0"))

        assert both < mixed < full

    def test_only_offloaded_blocks_cost_vram(self, tmp_path):
        model = mc_gguf.read(build_model(tmp_path, blocks=32))
        half = ctx.kv_bytes_per_token(model, ctx.Placement(gpu_layers=16))
        whole = ctx.kv_bytes_per_token(model, ctx.Placement())

        assert half == whole / 2

    def test_a_cpu_placement_costs_no_vram_at_all(self, tmp_path):
        model = build_model(tmp_path)
        estimate = ctx.estimate(model, ctx.Placement(on_gpu=False))

        assert estimate.total_bytes == 0
        assert estimate.kv_bytes_per_token == 0


class TestCapacity:
    def test_a_bigger_buffer_buys_proportionally_more_context(self, tmp_path):
        model = build_model(tmp_path)
        placement = ctx.Placement()

        one = ctx.capacity(model, placement, 1 * _GB)
        two = ctx.capacity(model, placement, 2 * _GB)

        assert two.theoretical == pytest.approx(one.theoretical * 2, rel=0.01)

    def test_the_recommendation_is_below_the_theoretical_figure(self, tmp_path):
        found = ctx.capacity(build_model(tmp_path), ctx.Placement(), 4 * _GB)

        assert 0 < found.recommended < found.theoretical
        assert found.recommended % ctx.CONTEXT_GRANULARITY == 0

    def test_the_model_ceiling_caps_what_is_offered(self, tmp_path):
        """Section 12: "If the model's own context limit is lower than the
        capacity implied by VRAM, the UI must cap usable context at the model
        limit"."""
        model = build_model(tmp_path, context=4096)
        found = ctx.capacity(model, ctx.Placement(), 32 * _GB)

        assert found.theoretical > 4096
        assert found.usable == 4096
        assert found.limited_by_model

    def test_a_model_with_no_ceiling_declared_is_not_capped_at_zero(self, tmp_path):
        model = build_model(tmp_path, context=0)
        found = ctx.capacity(model, ctx.Placement(), 4 * _GB)

        assert found.usable == found.recommended > 0

    def test_the_table_is_ordered_and_covers_the_offered_buffers(self, tmp_path):
        rows = ctx.table(build_model(tmp_path), ctx.Placement(), buffers_gb=(1, 2, 4))

        assert [row.budget_bytes for row in rows] == [1 * _GB, 2 * _GB, 4 * _GB]
        assert rows[0].theoretical < rows[1].theoretical < rows[2].theoretical

    def test_an_undescribable_model_reports_no_capacity_rather_than_guessing(self, tmp_path):
        path = tmp_path / "opaque.gguf"
        path.write_bytes(b"not a gguf at all")

        found = ctx.capacity(path, ctx.Placement(), 4 * _GB)

        assert (found.theoretical, found.recommended) == (0, 0)

    def test_automatic_sizing_never_returns_a_negative_buffer(self):
        """A model that does not fit is a decision for the broker; the
        estimator's job is only to say there is nothing spare."""
        assert ctx.automatic_buffer_bytes(4 * _GB, 8 * _GB, 1 * _GB) == 0


class TestEstimate:
    def test_the_three_budgets_stay_separate(self, tmp_path):
        """Section 11: weights, context/KV and runtime reserve are three
        numbers, and a total that cannot be decomposed is the bug."""
        estimate = ctx.estimate(build_model(tmp_path), ctx.Placement(context=8192))

        assert estimate.weights_bytes > 0
        assert estimate.kv_bytes > 0
        assert estimate.compute_bytes > 0
        assert (estimate.total_bytes
                == estimate.weights_bytes + estimate.kv_bytes + estimate.compute_bytes)
        assert estimate.resident_bytes == estimate.weights_bytes + estimate.kv_bytes

    def test_a_partial_offload_prorates_the_weights_and_says_so(self, tmp_path):
        model = build_model(tmp_path, blocks=32)
        whole = ctx.estimate(model, ctx.Placement())
        half = ctx.estimate(model, ctx.Placement(gpu_layers=16))

        assert half.weights_bytes == pytest.approx(whole.weights_bytes / 2, rel=0.01)
        assert "prorated" in half.detail

    def test_a_context_at_the_ceiling_is_flagged(self, tmp_path):
        estimate = ctx.estimate(build_model(tmp_path, context=4096),
                                ctx.Placement(context=4096))

        assert estimate.capped

    def test_an_unreadable_header_degrades_to_the_file_size(self, tmp_path):
        path = tmp_path / "opaque.gguf"
        path.write_bytes(b"x" * 4096)

        estimate = ctx.estimate(path, ctx.Placement())

        assert estimate.weights_bytes == 4096
        assert estimate.kv_bytes == 0
        assert "does not describe its attention shape" in estimate.detail


class TestCalibration:
    def test_an_estimate_starts_uncalibrated(self, tmp_path):
        assert not ctx.estimate(build_model(tmp_path), ctx.Placement()).calibrated

    def test_a_real_load_replaces_the_guessed_reserve(self, tmp_path):
        model = build_model(tmp_path)
        placement = ctx.Placement(context=8192)
        before = ctx.estimate(model, placement)
        observed = before.resident_bytes + 700 * _MB

        assert ctx.record_observation(model, placement, observed)

        after = ctx.estimate(model, placement)
        assert after.calibrated
        assert after.compute_bytes == pytest.approx(700 * _MB, rel=0.01)

    def test_calibration_survives_a_context_change(self, tmp_path):
        """The overhead is stored, not the total: a total would be wrong the
        moment the context moved, because the cache term scales with it."""
        model = build_model(tmp_path)
        placement = ctx.Placement(context=8192)
        ctx.record_observation(model, placement,
                               ctx.estimate(model, placement).resident_bytes + 500 * _MB)

        wider = ctx.estimate(model, placement.with_context(16384))

        assert wider.calibrated
        assert wider.compute_bytes == pytest.approx(500 * _MB, rel=0.01)
        assert wider.kv_bytes == 2 * ctx.estimate(model, placement).kv_bytes

    def test_a_different_placement_is_calibrated_separately(self, tmp_path):
        model = build_model(tmp_path)
        ctx.record_observation(model, ctx.Placement(context=8192),
                               ctx.estimate(model, ctx.Placement(context=8192)).resident_bytes
                               + 500 * _MB)

        other = ctx.estimate(model, ctx.Placement(context=8192, gpu_layers=16))

        assert not other.calibrated

    def test_an_implausible_measurement_is_discarded(self, tmp_path):
        """Smaller than the weights alone means something else freed memory
        during the load, not that the runtime is free."""
        model = build_model(tmp_path)
        placement = ctx.Placement()

        assert not ctx.record_observation(model, placement, 1024)
        assert not ctx.estimate(model, placement).calibrated


class TestPartialExpertResidency:
    """A mixture-of-experts model's experts, moved a few blocks at a time.

    The boolean this replaced made one choice available -- every expert in
    system RAM -- for a model that was two gigabytes too large, which is
    thirty-four blocks read over the PCIe bus to save a shortfall six would have
    covered. What is tested here is that the arithmetic in the middle is
    linear, that the two ends still mean what they meant, and that a calibration
    recorded at one split is never read back at another.
    """

    @pytest.fixture
    def moe(self):
        class Header:
            file_bytes = 40 * _GB
            block_count = 40
            usable = True
            context_length = 8192
            embedding_length = 3584
            expert_count = 8
            expert_used_count = 2
            mixture_of_experts = True
            expert_share = 0.80

        return Header()

    def test_nothing_moved_costs_the_whole_file(self, moe):
        placement = ctx.Placement(gpu_layers=ctx.ALL_LAYERS)

        assert ctx.expert_fraction_moved(moe, placement) == 0.0
        assert ctx.weights_bytes(moe, placement) == moe.file_bytes

    def test_every_expert_moved_leaves_only_what_is_not_an_expert(self, moe):
        placement = ctx.Placement(cpu_expert_layers=ctx.ALL_EXPERTS)

        assert ctx.expert_fraction_moved(moe, placement) == pytest.approx(0.80)
        assert ctx.weights_bytes(moe, placement) == pytest.approx(8 * _GB, rel=1e-6)

    def test_the_share_moved_is_prorated_across_the_blocks(self, moe):
        """Design intent section 7: ``expert_share * clamp(N / block_count)``."""
        placement = ctx.Placement(cpu_expert_layers=10)

        assert ctx.expert_fraction_moved(moe, placement) == pytest.approx(0.20)
        assert ctx.weights_bytes(moe, placement) == pytest.approx(32 * _GB, rel=1e-6)

    def test_each_further_block_costs_the_same_again(self, moe):
        """Linear, which is what lets the runtime solve for N rather than
        search for it."""
        sizes = [ctx.weights_bytes(moe, ctx.Placement(cpu_expert_layers=n))
                 for n in (0, 4, 8, 12)]
        steps = [before - after for before, after in zip(sizes, sizes[1:])]

        assert steps[0] == pytest.approx(steps[1], rel=1e-6)
        assert steps[1] == pytest.approx(steps[2], rel=1e-6)

    def test_more_blocks_than_the_model_has_moves_no_more_than_all_of_them(self, moe):
        assert ctx.expert_fraction_moved(moe, ctx.Placement(cpu_expert_layers=400)) == (
            pytest.approx(0.80))

    def test_a_dense_model_never_saves_anything_by_this(self, moe):
        class Dense(type(moe)):
            mixture_of_experts = False
            expert_share = 0.0

        assert ctx.expert_fraction_moved(Dense(), ctx.Placement(cpu_expert_layers=10)) == 0.0

    def test_a_header_with_no_block_count_discounts_nothing(self, moe):
        """An under-estimate ends in an out-of-memory error and an over-estimate
        ends in one avoidable eviction, so the unknown is charged for."""
        class Headerless(type(moe)):
            block_count = 0

        assert ctx.expert_fraction_moved(Headerless(),
                                         ctx.Placement(cpu_expert_layers=10)) == 0.0

    def test_the_boolean_helper_still_means_all_of_them(self, moe):
        placement = ctx.Placement().with_cpu_experts()

        assert placement.cpu_expert_layers == ctx.ALL_EXPERTS
        assert placement.cpu_experts is True
        assert placement.all_cpu_experts is True
        assert placement.with_cpu_experts(False).cpu_experts is False

    def test_a_partial_split_reads_as_experts_moved_but_not_all_of_them(self):
        placement = ctx.Placement(cpu_expert_layers=6)

        assert placement.cpu_experts is True
        assert placement.all_cpu_experts is False

    def test_the_calibration_key_carries_the_split(self):
        """Section 7: two placements that differ only in how many blocks left
        their experts behind are two different footprints."""
        keys = {ctx.Placement(cpu_expert_layers=n).key
                for n in (ctx.NO_EXPERTS, 8, 16, ctx.ALL_EXPERTS)}

        assert len(keys) == 4
        assert ctx.Placement(cpu_expert_layers=8).key.endswith("ncmoe-8")
        assert ctx.Placement(cpu_expert_layers=ctx.ALL_EXPERTS).key.endswith("cpu-moe")
        assert ctx.Placement().key.endswith("ncmoe-0")

    def test_a_measurement_at_one_split_is_not_read_back_at_another(self, tmp_path, moe,
                                                                    monkeypatch):
        model = build_model(tmp_path, blocks=40, size_mb=8)
        eight = ctx.Placement(cpu_expert_layers=8, context=4096)
        ten = ctx.Placement(cpu_expert_layers=10, context=4096)

        ctx.record_observation(model, eight, ctx.estimate(model, eight).total_bytes + 900 * _MB)

        assert ctx.estimate(model, eight).calibrated is True
        assert ctx.estimate(model, ten).calibrated is False

    def test_the_speed_token_names_the_placement(self):
        assert ctx.Placement().speed_token == "gpu"
        assert ctx.Placement(cpu_expert_layers=8).speed_token == "ncmoe-8"
        assert ctx.Placement(cpu_expert_layers=ctx.ALL_EXPERTS).speed_token == "cpu-moe"
        assert ctx.Placement(on_gpu=False).speed_token == "cpu"
        assert ctx.Placement(gpu_layers=ctx.NO_LAYERS).speed_token == "cpu"

    def test_a_partial_block_offload_is_its_own_speed(self):
        """Not one of the four the design intent lists, and reachable: once the
        experts have all gone and blocks start leaving, "on the GPU" covers
        placements an order of magnitude apart."""
        token = ctx.Placement(gpu_layers=20, cpu_expert_layers=ctx.ALL_EXPERTS).speed_token

        assert token == "cpu-moe-l20"
        assert token != ctx.Placement(cpu_expert_layers=ctx.ALL_EXPERTS).speed_token

    def test_the_placement_says_how_many_blocks_gave_their_experts_up(self):
        said = ctx.Placement(cpu_expert_layers=6).describe(40)

        assert "experts of 6 of 40 layers in system RAM" in said
        assert "all layers on the GPU" in said


class TestHybridModels:
    """A model whose blocks are not all the same shape.

    Multiplying the widest block by the block count overstates such a model's
    cache by a factor of two or more, which shows up as a context ceiling far
    below what the card can actually hold — and, before the header was read per
    block at all, as a TypeError on every reply.
    """

    @pytest.fixture
    def hybrid(self, tmp_path):
        """Eight blocks, four of which keep a cache."""
        metadata = b"".join([
            _text("general.architecture", "llama"),
            _u32("llama.block_count", 8),
            _u32("llama.context_length", 8192),
            _u32("llama.embedding_length", 4096),
            _u32("llama.attention.head_count", 32),
            _u32s("llama.attention.head_count_kv", [8, 0, 8, 0, 8, 0, 8, 0]),
        ])
        return mc_gguf.read(write_gguf(tmp_path / "hybrid.gguf", metadata, 6))

    def test_only_the_blocks_that_attend_are_paid_for(self, hybrid):
        per_token = ctx.kv_bytes_per_token(hybrid, ctx.Placement())

        assert per_token == 4 * 8 * (128 + 128) * 2.0

    def test_a_partial_offload_costs_the_blocks_llama_cpp_puts_on_the_card(self, hybrid):
        """llama.cpp offloads the *last* n layers, and in this model the last
        two are one that attends and one that does not."""
        per_token = ctx.kv_bytes_per_token(hybrid, ctx.Placement(gpu_layers=2))

        assert per_token == 1 * 8 * (128 + 128) * 2.0

    def test_a_capacity_estimate_is_produced_rather_than_an_exception(self, hybrid):
        found = ctx.capacity(hybrid.path, ctx.Placement(), 4 * _GB, gguf=hybrid)

        assert found.usable > 0
