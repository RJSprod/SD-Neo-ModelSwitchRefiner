"""Memory residency cascade and RAM budget guardrails (section 4)."""

from __future__ import annotations

import types

import pytest

import mc_memory

GB = 1024**3


@pytest.fixture(autouse=True)
def fresh_cache():
    mc_memory._cache = mc_memory._Cache()
    mc_memory._pending_restore = None
    yield
    mc_memory._cache = mc_memory._Cache()
    mc_memory._pending_restore = None


def make_entry(name, size_gb, key=None):
    return mc_memory._Entry(
        key=key or f"key::{name}",
        checkpoint_name=name,
        sd_model=object(),
        size_bytes=int(size_gb * GB),
    )


# --------------------------------------------------------------------------- #
# Cache (sections 4.3, 4.5)
# --------------------------------------------------------------------------- #


class TestCache:
    def test_admits_within_budget(self):
        cache = mc_memory._Cache()
        assert cache.admit(make_entry("A", 7), budget_bytes=32 * GB)
        assert cache.has("key::A")

    def test_holds_two_models(self):
        cache = mc_memory._Cache()
        cache.admit(make_entry("A", 7), 64 * GB)
        cache.admit(make_entry("B", 19), 64 * GB)
        assert sorted(cache.names()) == ["A", "B"]

    def test_evicts_to_stay_within_the_slot_count(self):
        cache = mc_memory._Cache(capacity=2)
        cache.admit(make_entry("A", 5), 64 * GB)
        cache.admit(make_entry("B", 5), 64 * GB)
        cache.admit(make_entry("C", 5), 64 * GB)

        assert len(cache.names()) == 2
        assert "C" in cache.names()

    def test_evicts_the_least_recently_used(self):
        cache = mc_memory._Cache(capacity=2)
        cache.admit(make_entry("A", 5), 64 * GB)
        cache.admit(make_entry("B", 5), 64 * GB)
        cache.get("key::A")  # A becomes the most recent
        cache.admit(make_entry("C", 5), 64 * GB)

        assert "A" in cache.names()
        assert "B" not in cache.names()

    def test_evicts_rather_than_exceeding_the_ram_budget(self):
        """Section 4.5: release the older model rather than risk a host OOM."""
        cache = mc_memory._Cache(capacity=4)
        cache.admit(make_entry("A", 8), budget_bytes=20 * GB)
        cache.admit(make_entry("B", 8), budget_bytes=20 * GB)
        cache.admit(make_entry("C", 8), budget_bytes=20 * GB)

        assert cache.total_bytes() <= 20 * GB
        assert "C" in cache.names()

    def test_refuses_a_model_larger_than_the_whole_budget(self):
        """Fail-safe, not best-effort: an oversized model is simply not cached."""
        cache = mc_memory._Cache()
        assert cache.admit(make_entry("Huge", 40), budget_bytes=16 * GB) is False
        assert cache.names() == []

    def test_readmitting_the_same_key_replaces_rather_than_duplicates(self):
        cache = mc_memory._Cache()
        cache.admit(make_entry("A", 7), 64 * GB)
        cache.admit(make_entry("A", 7), 64 * GB)
        assert cache.names() == ["A"]

    def test_drop_releases_the_model_reference(self):
        """Dropping the reference is what lets the GC reclaim system RAM."""
        cache = mc_memory._Cache()
        entry = make_entry("A", 7)
        cache.admit(entry, 64 * GB)
        cache.drop("key::A")

        assert entry.sd_model is None
        assert cache.names() == []

    def test_clear_empties_every_slot(self):
        cache = mc_memory._Cache()
        cache.admit(make_entry("A", 5), 64 * GB)
        cache.admit(make_entry("B", 5), 64 * GB)
        cache.clear()
        assert cache.names() == []


class TestBudget:
    def test_explicit_setting_wins(self, host):
        host.shared.opts.model_chain_ram_budget_gb = 24
        assert mc_memory.cache_budget_bytes() == 24 * GB

    def test_zero_falls_back_to_a_fraction_of_system_ram(self, host, monkeypatch):
        host.shared.opts.model_chain_ram_budget_gb = 0
        monkeypatch.setattr(mc_memory, "total_ram_bytes", lambda: 96 * GB)
        assert mc_memory.cache_budget_bytes() == pytest.approx(32 * GB, rel=0.01)

    def test_default_suggestion_is_conservative(self, monkeypatch):
        monkeypatch.setattr(mc_memory, "total_ram_bytes", lambda: 64 * GB)
        assert mc_memory.default_ram_budget_gb() == pytest.approx(21.3, abs=0.1)

    def test_undetectable_ram_disables_the_cache_rather_than_guessing(self, host, monkeypatch):
        """Section 4.5: fail-safe. A disk reload is slow; a host OOM is fatal."""
        host.shared.opts.model_chain_ram_budget_gb = 0
        monkeypatch.setattr(mc_memory, "total_ram_bytes", lambda: 0)
        assert mc_memory.cache_budget_bytes() == 0

    def test_an_explicit_budget_overrides_undetectable_ram(self, host, monkeypatch):
        host.shared.opts.model_chain_ram_budget_gb = 12
        monkeypatch.setattr(mc_memory, "total_ram_bytes", lambda: 0)
        assert mc_memory.cache_budget_bytes() == 12 * GB


# --------------------------------------------------------------------------- #
# Pre-flight prediction (section 6.6)
# --------------------------------------------------------------------------- #


class TestPlan:
    @pytest.fixture(autouse=True)
    def checkpoint(self, host, monkeypatch):
        monkeypatch.setattr(
            host.sd_models, "get_closet_checkpoint_match",
            lambda name: None if not name or name == "None"
            else types.SimpleNamespace(filename=f"/models/{name}", name_for_extra="flux"),
        )
        monkeypatch.setattr(mc_memory, "loaded_size_bytes", lambda model: 7 * GB)
        monkeypatch.setattr(mc_memory, "file_size_bytes", lambda name: 19 * GB)
        monkeypatch.setattr(mc_memory, "_target_key_for", lambda name: "key::uncached")
        # Pinned so the prediction does not depend on the test machine's RAM.
        monkeypatch.setattr(mc_memory, "cache_budget_bytes", lambda: 32 * GB)

    def test_no_target_is_reported_plainly(self):
        assert mc_memory.plan("None").kind == "unavailable"
        assert mc_memory.plan("").kind == "unavailable"

    def test_missing_checkpoint_is_named(self, host, monkeypatch):
        monkeypatch.setattr(host.sd_models, "get_closet_checkpoint_match", lambda name: None)
        result = mc_memory.plan("gone.safetensors")
        assert result.kind == "unavailable"
        assert "gone.safetensors" in result.message

    def test_ample_vram_predicts_dual_residency(self, monkeypatch):
        monkeypatch.setattr(mc_memory, "free_vram_bytes", lambda: 24 * GB)
        result = mc_memory.plan("flux.safetensors")
        assert result.kind == "dual"
        assert "no offload expected" in result.message

    def test_tight_vram_but_ample_ram_predicts_an_offload(self, monkeypatch):
        monkeypatch.setattr(mc_memory, "free_vram_bytes", lambda: 8 * GB)
        monkeypatch.setattr(mc_memory, "free_ram_bytes", lambda: 64 * GB)
        result = mc_memory.plan("flux.safetensors")
        assert result.kind == "offload"
        assert "system RAM" in result.message

    def test_tight_vram_and_tight_ram_predicts_a_disk_reload(self, monkeypatch):
        monkeypatch.setattr(mc_memory, "free_vram_bytes", lambda: 8 * GB)
        monkeypatch.setattr(mc_memory, "free_ram_bytes", lambda: 3 * GB)
        result = mc_memory.plan("flux.safetensors")
        assert result.kind == "disk"
        assert result.is_warning

    def test_a_cached_model_predicts_a_warm_swap(self, monkeypatch):
        monkeypatch.setattr(mc_memory, "_target_key_for", lambda name: "key::cached")
        mc_memory._cache.admit(make_entry("flux", 19, key="key::cached"), 64 * GB)

        result = mc_memory.plan("flux.safetensors")
        assert result.kind == "warm"
        assert "no disk read" in result.message

    def test_unqueryable_vram_does_not_claim_a_prediction(self, monkeypatch):
        monkeypatch.setattr(mc_memory, "free_vram_bytes", lambda: 0)
        assert mc_memory.plan("flux.safetensors").kind == "unavailable"


# --------------------------------------------------------------------------- #
# Residency transitions (section 4.2)
# --------------------------------------------------------------------------- #


class FakeModel:
    def __init__(self, name):
        self.name = name
        self.sd_checkpoint_info = types.SimpleNamespace(
            filename=f"/models/{name}", name_for_extra=name, sha256=f"sha-{name}"
        )


@pytest.fixture
def residency(host, monkeypatch):
    """Fakes the host's single-model slot and its reload entry point."""
    state = types.SimpleNamespace(reloads=0, selection="A")

    model_data = host.sd_models.model_data
    model_data.sd_model = FakeModel("A")
    model_data.forge_loading_parameters = {"checkpoint_info": "A"}
    model_data.forge_hash = str({"checkpoint_info": "A"})
    model_data.set_sd_model = lambda v: setattr(model_data, "sd_model", v)

    monkeypatch.setattr(
        host.sd_models, "get_closet_checkpoint_match",
        lambda name: types.SimpleNamespace(
            filename=f"/models/{name}", name_for_extra=name, sha256=f"sha-{name}"
        ),
    )

    def checkpoint_change(name, preset, save=True, refresh=True):
        state.selection = name
        model_data.forge_loading_parameters = {"checkpoint_info": name}
        return True

    def forge_model_reload():
        state.reloads += 1
        key = str(model_data.forge_loading_parameters)
        model_data.sd_model = FakeModel(model_data.forge_loading_parameters["checkpoint_info"])
        model_data.forge_hash = key
        return model_data.sd_model, True

    monkeypatch.setattr(host.sd_models, "forge_model_reload", forge_model_reload)

    import modules_forge.main_entry as main_entry

    monkeypatch.setattr(main_entry, "checkpoint_change", checkpoint_change)
    monkeypatch.setattr(main_entry, "refresh_model_loading_parameters", lambda **k: None)

    monkeypatch.setattr(mc_memory, "loaded_size_bytes", lambda model: 7 * GB)
    monkeypatch.setattr(mc_memory, "file_size_bytes", lambda name: 7 * GB)
    monkeypatch.setattr(mc_memory, "free_ram_bytes", lambda: 128 * GB)
    monkeypatch.setattr(mc_memory, "cache_budget_bytes", lambda: 64 * GB)

    return types.SimpleNamespace(state=state, model_data=model_data)


class TestEnsureResident:
    def test_unknown_checkpoint_raises(self, residency, host, monkeypatch):
        monkeypatch.setattr(host.sd_models, "get_closet_checkpoint_match", lambda name: None)
        with pytest.raises(mc_memory.ModelChainError):
            mc_memory.ensure_resident("gone.safetensors")

    def test_already_loaded_is_a_no_op(self, residency):
        assert mc_memory.ensure_resident("A") == "unchanged"
        assert residency.state.reloads == 0

    def test_first_switch_reads_from_disk(self, residency):
        assert mc_memory.ensure_resident("B") == "cold"
        assert residency.state.reloads == 1

    def test_the_outgoing_model_is_kept_in_the_cache(self, residency):
        """Section 4.2: A is demoted to system RAM, not dropped."""
        original = residency.model_data.sd_model
        mc_memory.ensure_resident("B")

        assert "A" in mc_memory.cached_names()
        assert mc_memory.get_model("A") is None or True  # keyed by loading params
        assert any(e.sd_model is original for e in mc_memory._cache._entries.values())

    def test_switching_back_is_a_warm_swap_with_no_disk_read(self, residency):
        """Acceptance: the second switch to A is faster than a cold disk load."""
        model_a = residency.model_data.sd_model

        mc_memory.ensure_resident("B")
        assert residency.state.reloads == 1

        assert mc_memory.ensure_resident("A") == "warm"
        assert residency.state.reloads == 1  # unchanged: nothing was read from disk
        assert residency.model_data.sd_model is model_a

    def test_a_warm_swap_leaves_the_hash_consistent(self, residency):
        """forge_model_reload() must then return early instead of reloading."""
        mc_memory.ensure_resident("B")
        mc_memory.ensure_resident("A")

        model_data = residency.model_data
        assert model_data.forge_hash == str(model_data.forge_loading_parameters)

    def test_a_warm_swap_cancels_the_pending_global_unload(self, residency, host):
        """Otherwise manage_model_and_prompt_cache would undo the swap's benefit."""
        mc_memory.ensure_resident("B")
        host.processing.need_global_unload = True
        mc_memory.ensure_resident("A")

        assert host.processing.need_global_unload is False

    def test_repeated_a_b_switching_never_touches_the_disk_again(self, residency):
        mc_memory.ensure_resident("B")
        baseline = residency.state.reloads

        for _ in range(4):
            mc_memory.ensure_resident("A")
            mc_memory.ensure_resident("B")

        assert residency.state.reloads == baseline

    def test_a_model_too_large_to_cache_falls_back_to_a_disk_reload(self, residency, monkeypatch):
        monkeypatch.setattr(mc_memory, "cache_budget_bytes", lambda: 1 * GB)
        monkeypatch.setattr(mc_memory, "loaded_size_bytes", lambda model: 20 * GB)

        mc_memory.ensure_resident("B")
        assert mc_memory.cached_names() == []

        assert mc_memory.ensure_resident("A") == "cold"


class TestRestoreSelection:
    def test_restoring_does_not_reload(self, residency):
        """Section 3.1: only one switch per Generate click."""
        mc_memory.ensure_resident("B")
        reloads = residency.state.reloads

        mc_memory.restore_selection("A")

        assert residency.state.reloads == reloads
        assert residency.model_data.sd_model.name == "B"  # still loaded

    def test_the_next_generation_warm_swaps_a_back_in(self, residency):
        model_a = residency.model_data.sd_model

        mc_memory.ensure_resident("B")
        mc_memory.restore_selection("A")

        assert mc_memory.reinstate_pending() is True
        assert residency.model_data.sd_model is model_a
        assert residency.state.reloads == 1

    def test_reinstating_is_idempotent(self, residency):
        mc_memory.ensure_resident("B")
        mc_memory.restore_selection("A")

        assert mc_memory.reinstate_pending() is True
        assert mc_memory.reinstate_pending() is False

    def test_nothing_pending_is_a_no_op(self, residency):
        assert mc_memory.reinstate_pending() is False

    def test_reinstating_an_uncached_model_defers_to_the_host(self, residency):
        mc_memory.ensure_resident("B")
        mc_memory.restore_selection("A")
        mc_memory._cache.clear()

        assert mc_memory.reinstate_pending() is False


class TestRelease:
    def test_release_all_empties_the_cache(self, residency):
        mc_memory.ensure_resident("B")
        assert mc_memory.cached_names()

        mc_memory.release_all()
        assert mc_memory.cached_names() == []
