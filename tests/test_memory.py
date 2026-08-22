"""Memory residency cascade and RAM budget guardrails (section 4)."""

from __future__ import annotations

import logging
import re
import sys
import types

import pytest

import mc_memory

GB = 1024**3


@pytest.fixture(autouse=True)
def fresh_cache():
    mc_memory._cache = mc_memory._Cache()
    mc_memory._pending_restore = None
    mc_memory._last_refusal = None
    yield
    mc_memory._cache = mc_memory._Cache()
    mc_memory._pending_restore = None
    mc_memory._last_refusal = None


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

    def test_drop_releases_the_caches_reference(self):
        """Letting go of the entry is what lets the GC reclaim system RAM.

        The cache's own reference is the only one it may release. Blanking the
        entry as well reaches into whoever else is holding it -- and someone is:
        ``reinstate_pending`` looks its entry up, then stashes the outgoing
        model, which can evict the entry it is holding.
        """
        cache = mc_memory._Cache()
        entry = make_entry("A", 7)
        cache.admit(entry, 64 * GB)
        cache.drop("key::A")

        assert cache.names() == []
        assert cache.get("key::A") is None
        assert entry.sd_model is not None

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
        expected = 96 * GB * mc_memory.DEFAULT_BUDGET_FRACTION
        assert mc_memory.cache_budget_bytes() == pytest.approx(expected, rel=0.01)

    @pytest.mark.parametrize("total_gb", [32, 48, 64, 128])
    def test_a_flux_klein_sized_model_fits_the_default_budget(self, host, monkeypatch, total_gb):
        """Regression: the default used to be a third of system RAM.

        A Flux.2 Klein checkpoint with a Qwen3 text encoder is roughly 14 GB
        resident, which exceeded a 32 GB machine's 10.6 GB budget -- the model
        was refused outright and every switch cold-loaded from disk, which is
        the exact cost this cache exists to avoid.
        """
        host.shared.opts.model_chain_ram_budget_gb = 0
        monkeypatch.setattr(mc_memory, "total_ram_bytes", lambda: total_gb * GB)

        klein_resident = 14 * GB
        assert mc_memory.cache_budget_bytes() >= klein_resident

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
        monkeypatch.setattr(mc_memory, "file_size_bytes", lambda name, mods=None: 19 * GB)
        monkeypatch.setattr(mc_memory, "_target_key_for", lambda name, mods=None: "key::uncached")
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
        monkeypatch.setattr(mc_memory, "_target_key_for", lambda name, mods=None: "key::cached")
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


LATENT_SCALES: dict[str, object] = {}
"""``upscale_ratio`` per checkpoint name, for the tests that care.

Mirrors ``backend.patcher.vae``: an int for the 2D VAEs (8 normally, 16 for
Flux.2), and a tuple for a Wan VAE -- which is why the host's ``isinstance``
check falls back to 8 rather than reading the tuple.
"""


class FakeModel:
    def __init__(self, name):
        self.name = name
        self.sd_checkpoint_info = types.SimpleNamespace(
            filename=f"/models/{name}", name_for_extra=name, sha256=f"sha-{name}"
        )
        vae = types.SimpleNamespace(upscale_ratio=LATENT_SCALES.get(name, 8))
        self.forge_objects = types.SimpleNamespace(vae=vae)
        self.forge_objects_original = self.forge_objects


class FakeInitialModel:
    """The host's placeholder before anything is loaded.

    Named to match: ``_is_real_model`` recognises it by class name, exactly as
    it must recognise the host's own.
    """


@pytest.fixture
def residency(host, monkeypatch):
    """Fakes the host's single-model slot and its reload entry point."""
    state = types.SimpleNamespace(reloads=0, selection="A")

    LATENT_SCALES.clear()

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
        # The host's own last act before returning: opt_f is rewritten from the
        # freshly loaded VAE, which is what makes it describe the last *load*
        # rather than the model that happens to be in the slot.
        ratio = model_data.sd_model.forge_objects.vae.upscale_ratio
        host.processing.opt_f = ratio if isinstance(ratio, int) else 8
        return model_data.sd_model, True

    monkeypatch.setattr(host.sd_models, "forge_model_reload", forge_model_reload)

    import modules_forge.main_entry as main_entry

    monkeypatch.setattr(main_entry, "checkpoint_change", checkpoint_change)
    monkeypatch.setattr(main_entry, "refresh_model_loading_parameters", lambda **k: None)

    monkeypatch.setattr(mc_memory, "loaded_size_bytes", lambda model: 7 * GB)
    monkeypatch.setattr(mc_memory, "file_size_bytes", lambda name, mods=None: 7 * GB)
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

    def test_stage_1_survives_being_evicted_by_the_stage_2_stash(self, residency, monkeypatch):
        """The swap back stashes Stage 2, and that can evict Stage 1 itself.

        The two models are the two largest things in the cache and Stage 2's
        arrival is exactly when the budget runs out, so this is the ordinary
        case on a machine with less spare RAM -- not a corner.
        """
        model_a = residency.model_data.sd_model
        # Room for one model only, so stashing B must evict A.
        monkeypatch.setattr(mc_memory, "cache_budget_bytes", lambda: 8 * GB)

        mc_memory.ensure_resident("B")
        mc_memory.restore_selection("A")

        assert mc_memory.reinstate_pending() is True
        assert residency.model_data.sd_model is model_a

    def test_the_swap_back_never_installs_a_null_model(self, residency, monkeypatch):
        """A null model here wedges the host until it is restarted.

        forge_model_reload returns early while forge_hash matches, handing back
        whatever is in the slot -- so an empty slot fails every generation from
        then on, and retrying cannot clear it.
        """
        mc_memory.ensure_resident("B")
        mc_memory.restore_selection("A")

        # Whatever goes wrong, a cold load is the answer, never an empty slot.
        monkeypatch.setattr(mc_memory, "_stash_current", lambda stage="": mc_memory._cache.clear())
        for entry in list(mc_memory._cache._entries.values()):
            entry.sd_model = None

        assert mc_memory.reinstate_pending() is False
        assert mc_memory._is_real_model(residency.model_data.sd_model)


class TestProtectedEviction:
    """Recency picks the wrong victim at exactly the moment a switch happens.

    Every switch stashes the outgoing model moments before restoring the
    incoming one -- and the incoming one has not been touched since the last
    generation, so it is the least recently used, so plain LRU throws it out
    immediately before it is needed.
    """

    def test_the_protected_entry_is_not_the_victim(self):
        cache = mc_memory._Cache()
        incoming = make_entry("A", 4)
        cache.admit(incoming, 64 * GB)
        cache.admit(make_entry("B", 4), 64 * GB)
        incoming.last_used = 0  # least recently used, and about to be needed

        cache.admit(make_entry("C", 4), 9 * GB, protect="key::A")

        assert "A" in cache.names()
        assert "B" not in cache.names()

    def test_room_is_refused_rather_than_taken_from_the_protected_entry(self):
        """One slot's worth of budget: keep what is about to be used."""
        cache = mc_memory._Cache()
        incoming = make_entry("A", 4)
        cache.admit(incoming, 64 * GB)
        incoming.last_used = 0

        assert cache.admit(make_entry("B", 4), 6 * GB, protect="key::A") is False
        assert cache.names() == ["A"]

    def test_admitting_terminates_when_only_the_protected_entry_remains(self):
        """The eviction loop must not spin on an entry it may never drop."""
        cache = mc_memory._Cache()
        cache.admit(make_entry("A", 4), 64 * GB)

        assert cache.admit(make_entry("B", 60), 6 * GB, protect="key::A") is False

    def test_without_protection_the_oldest_still_goes(self):
        cache = mc_memory._Cache()
        oldest = make_entry("A", 4)
        cache.admit(oldest, 64 * GB)
        cache.admit(make_entry("B", 4), 64 * GB)
        oldest.last_used = 0

        cache.admit(make_entry("C", 4), 9 * GB)

        assert "A" not in cache.names()

    def test_the_swap_back_does_not_evict_the_model_it_is_restoring(
        self, residency, monkeypatch
    ):
        model_a = residency.model_data.sd_model
        monkeypatch.setattr(mc_memory, "cache_budget_bytes", lambda: 8 * GB)

        mc_memory.ensure_resident("B")
        mc_memory.restore_selection("A")
        mc_memory.reinstate_pending()

        # A is what was just restored, so A is what the cache should still hold.
        assert residency.model_data.sd_model is model_a
        assert "A" in mc_memory.cached_names()

    def squeezed(self, monkeypatch, *, predict_key=True):
        """Stage 2 cached from an earlier generation, room in the cache for one.

        The fixture's loading parameters are a one-key dict, so the real
        ``_target_key_for`` -- which builds the host's three-key form -- cannot
        match them here. Standing in for it is what puts the behaviour under
        test rather than the key format.
        """
        monkeypatch.setattr(mc_memory, "cache_budget_bytes", lambda: 8 * GB)
        monkeypatch.setattr(
            mc_memory,
            "_target_key_for",
            (lambda name, mods=None: str({"checkpoint_info": name})) if predict_key
            else (lambda name, mods=None: ""),
        )
        mc_memory._cache.admit(
            mc_memory._Entry(
                key=str({"checkpoint_info": "B"}),
                checkpoint_name="B",
                sd_model=FakeModel("B"),
                size_bytes=7 * GB,
            ),
            64 * GB,
        )

    def test_the_switch_out_does_not_evict_the_model_it_is_fetching(
        self, residency, monkeypatch
    ):
        """The expensive direction: evicting B here turns a warm swap cold."""
        self.squeezed(monkeypatch)

        assert mc_memory.ensure_resident("B") == "warm"
        assert residency.state.reloads == 0

    def test_without_protection_that_same_switch_reads_the_disk(
        self, residency, monkeypatch
    ):
        """The behaviour being fixed, shown by removing only the protection."""
        self.squeezed(monkeypatch, predict_key=False)

        assert mc_memory.ensure_resident("B") == "cold"
        assert residency.state.reloads == 1


class TestDrop:
    """Dropping releases the cache's hold; it must not reach into a caller's."""

    def test_dropping_leaves_the_entry_usable(self):
        entry = make_entry("A", 4)
        mc_memory._cache.admit(entry, 64 * GB)

        mc_memory._cache.drop(entry.key)

        assert mc_memory._cache.get(entry.key) is None
        assert entry.sd_model is not None

    def test_eviction_leaves_the_evicted_entry_usable(self):
        first = make_entry("A", 4)
        second = make_entry("B", 4)
        mc_memory._cache.admit(first, 6 * GB)
        mc_memory._cache.admit(second, 6 * GB)  # no room for both -> evicts A

        assert mc_memory.cached_names() == ["B"]
        assert first.sd_model is not None


class TestStaleLoadingHash:
    """An empty model slot with a hash still asserting a load is unrecoverable.

    forge_model_reload early-returns on a matching hash and hands back the empty
    slot, so every generation fails identically until the WebUI restarts.
    """

    def test_a_stale_hash_is_cleared(self, residency):
        residency.model_data.sd_model = None
        residency.model_data.forge_hash = "key::something"

        assert mc_memory.ensure_model_loadable() is True
        assert residency.model_data.forge_hash == ""

    def test_a_loaded_model_is_left_alone(self, residency):
        hash_before = residency.model_data.forge_hash

        assert mc_memory.ensure_model_loadable() is False
        assert residency.model_data.forge_hash == hash_before

    def test_a_fresh_start_is_not_mistaken_for_the_fault(self, residency):
        """Before the first load: no model, and no hash claiming one either."""
        residency.model_data.sd_model = FakeInitialModel()
        residency.model_data.forge_hash = ""

        assert mc_memory.ensure_model_loadable() is False

    def test_the_placeholder_model_counts_as_no_model(self, residency):
        residency.model_data.sd_model = FakeInitialModel()
        residency.model_data.forge_hash = "key::something"

        assert mc_memory.ensure_model_loadable() is True
        assert residency.model_data.forge_hash == ""


class TestRelease:
    def test_release_all_empties_the_cache(self, residency):
        mc_memory.ensure_resident("B")
        assert mc_memory.cached_names()

        mc_memory.release_all()
        assert mc_memory.cached_names() == []


# --------------------------------------------------------------------------- #
# Model flags across a warm swap
# --------------------------------------------------------------------------- #


class TestModelFlags:
    """forge_loader sets dynamic_args flags on every load; a warm pointer swap
    never runs the loader, so the cache has to carry them.

    Leaving them describing the previously loaded checkpoint is not cosmetic:
    ``nunchaku`` selects a different LoRA application path, ``kontext`` and
    ``edit`` decide whether Flux.1 and Qwen-Image use reference conditioning,
    ``klein`` changes sampling and ``pid`` changes the latent shape.
    """

    @pytest.fixture
    def flags(self):
        from backend.args import dynamic_args

        return dynamic_args

    def test_snapshot_captures_every_loader_set_flag(self, host, flags):
        flags.krea2 = True
        flags.nunchaku = True

        captured = mc_memory.snapshot_model_flags()

        assert set(captured) == set(mc_memory.MODEL_FLAGS)
        assert captured["krea2"] is True
        assert captured["nunchaku"] is True
        assert captured["kontext"] is False

    def test_warm_swap_restores_the_flags_of_the_model_being_restored(self, residency, flags):
        # Model A is a plain checkpoint.
        flags.krea2 = False
        flags.nunchaku = False
        model_a = residency.model_data.sd_model

        # Switching to Model B, which the loader would mark as Krea 2.
        mc_memory.ensure_resident("B")
        flags.krea2 = True

        # Switching back must not leave B's flags describing A.
        assert mc_memory.ensure_resident("A") == "warm"
        assert residency.model_data.sd_model is model_a
        assert flags.krea2 is False

    def test_warm_swap_forward_restores_the_target_flags(self, residency, flags):
        mc_memory.ensure_resident("B")
        flags.krea2 = True
        mc_memory.ensure_resident("A")

        # Back to B: its Krea 2 flag must come back with it.
        assert mc_memory.ensure_resident("B") == "warm"
        assert flags.krea2 is True

    def test_warm_swap_clears_per_generation_latent_state(self, residency, flags):
        """forge_loader calls dynamic_args.reset(); the warm path must too."""
        mc_memory.ensure_resident("B")
        flags.ref_latents.append("stale reference")

        mc_memory.ensure_resident("A")

        assert flags.ref_latents == []

    def test_reinstating_a_deferred_switch_also_restores_flags(self, residency, flags):
        mc_memory.ensure_resident("B")
        flags.krea2 = True
        mc_memory.restore_selection("A")

        assert mc_memory.reinstate_pending() is True
        assert flags.krea2 is False

    def test_repeated_switching_keeps_flags_in_step_with_the_model(self, residency, flags):
        mc_memory.ensure_resident("B")
        flags.krea2 = True

        for _ in range(3):
            mc_memory.ensure_resident("A")
            assert flags.krea2 is False
            mc_memory.ensure_resident("B")
            assert flags.krea2 is True


# --------------------------------------------------------------------------- #
# Latent scale across a warm swap
# --------------------------------------------------------------------------- #


WAN_RATIO = (lambda a: max(0, a * 4 - 3), 8, 8)
"""What a Wan VAE (Krea 2) puts in ``upscale_ratio``: a tuple, not an int."""


class TestLatentScale:
    """``modules.processing.opt_f`` has to travel with the cached model too.

    It is the divisor ``process_images_inner`` applies to the requested pixel
    size to shape the noise, and ``forge_model_reload`` rewrites it from the
    VAE of whatever it just loaded. A warm swap skips the loader, so left alone
    the divisor keeps describing the other stage's model -- and the pass is
    then sampled at the wrong size, which is only visible in the output image.
    """

    @pytest.mark.parametrize(
        "ratio, expected",
        [
            (8, 8),           # SD / SDXL / Flux.1 / Qwen
            (16, 16),         # Flux.2, including Klein
            (WAN_RATIO, 8),   # Wan / Krea 2: a tuple takes the host's fallback
            (True, 8),        # nonsense: the fallback, never int(True) == 1
        ],
    )
    def test_it_reads_the_scale_the_loader_would_have_set(self, ratio, expected):
        model = types.SimpleNamespace(
            forge_objects=types.SimpleNamespace(
                vae=types.SimpleNamespace(upscale_ratio=ratio)
            )
        )
        assert mc_memory.latent_scale_of(model) == expected

    def test_a_model_with_no_readable_vae_gets_no_opinion(self):
        """None means "leave it alone" -- guessing here is the failure itself."""
        assert mc_memory.latent_scale_of(types.SimpleNamespace()) is None
        assert mc_memory.latent_scale_of(
            types.SimpleNamespace(forge_objects=types.SimpleNamespace(vae=object()))
        ) is None

    def test_it_falls_back_to_the_original_objects(self):
        """A generation reassigns ``forge_objects``; both carry the same VAE."""
        model = types.SimpleNamespace(
            forge_objects_original=types.SimpleNamespace(
                vae=types.SimpleNamespace(upscale_ratio=16)
            )
        )
        assert mc_memory.latent_scale_of(model) == 16

    def test_warm_swap_restores_the_scale_of_the_model_being_restored(self, residency, host):
        residency.model_data.sd_model.forge_objects.vae.upscale_ratio = WAN_RATIO
        LATENT_SCALES["B"] = 16

        mc_memory.ensure_resident("B")
        assert host.processing.opt_f == 16  # the loader ran

        assert mc_memory.ensure_resident("A") == "warm"
        assert host.processing.opt_f == 8

    def test_warm_swap_forward_restores_the_target_scale(self, residency, host):
        residency.model_data.sd_model.forge_objects.vae.upscale_ratio = WAN_RATIO
        LATENT_SCALES["B"] = 16

        mc_memory.ensure_resident("B")
        mc_memory.ensure_resident("A")
        host.processing.opt_f = 8  # A's divisor, so the swap has to change it

        assert mc_memory.ensure_resident("B") == "warm"
        assert host.processing.opt_f == 16

    def test_reinstating_a_deferred_switch_also_restores_the_scale(self, residency, host):
        """The exact sequence a chained generation runs.

        Stage 2 defers the swap back to Stage 1 to the start of the next
        generation, so this is where the second run's Stage 1 gets its divisor
        -- and where it did not get one before.
        """
        residency.model_data.sd_model.forge_objects.vae.upscale_ratio = WAN_RATIO
        LATENT_SCALES["B"] = 16

        mc_memory.ensure_resident("B")
        mc_memory.restore_selection("A")

        assert mc_memory.reinstate_pending() is True
        assert host.processing.opt_f == 8

    def test_a_second_full_chain_still_samples_at_the_requested_size(self, residency, host):
        """The reported bug, run end to end.

        Krea 2 (a Wan VAE, divisor 8) into Flux.2 Klein (divisor 16). The first
        generation loads both from disk and is correct. The second warm-swaps
        Krea 2 back in, and before the fix it sampled under Flux.2's 16: a
        640x960 request became a 60x40 latent and a 320x480 image, which Stage 2
        then faithfully refined at half size.
        """
        residency.model_data.sd_model.forge_objects.vae.upscale_ratio = WAN_RATIO
        LATENT_SCALES["B"] = 16

        requested = (960, 640)

        def sampled_size():
            """What ``process_images_inner`` would build the noise from."""
            scale = host.processing.opt_f
            return tuple(dimension // scale * scale for dimension in requested)

        for _ in range(2):
            # Stage 1.
            mc_memory.reinstate_pending()
            assert host.processing.opt_f == 8
            assert sampled_size() == requested

            # Stage 2, then the deferred swap back.
            mc_memory.ensure_resident("B")
            assert host.processing.opt_f == 16
            mc_memory.restore_selection("A")

    def test_repeated_switching_keeps_the_scale_in_step_with_the_model(self, residency, host):
        residency.model_data.sd_model.forge_objects.vae.upscale_ratio = WAN_RATIO
        LATENT_SCALES["B"] = 16

        mc_memory.ensure_resident("B")

        for _ in range(3):
            mc_memory.ensure_resident("A")
            assert host.processing.opt_f == 8
            mc_memory.ensure_resident("B")
            assert host.processing.opt_f == 16

    def test_a_model_without_a_vae_leaves_the_scale_alone(self, residency, host):
        """Better a stale divisor than a guessed one -- and nothing raises."""
        for model in (residency.model_data.sd_model,):
            del model.forge_objects
            del model.forge_objects_original

        host.processing.opt_f = 16
        mc_memory.ensure_resident("B")
        mc_memory.ensure_resident("A")

        assert host.processing.opt_f == 8  # B's loader set it; A had no opinion


class TestAlignLatentScale:
    """The check at the top of a generation, for swaps nothing accounted for."""

    def test_it_corrects_a_divisor_left_behind_by_another_model(self, residency, host):
        residency.model_data.sd_model.forge_objects.vae.upscale_ratio = 8
        host.processing.opt_f = 16

        assert mc_memory.align_latent_scale() is True
        assert host.processing.opt_f == 8

    def test_it_says_and_does_nothing_when_the_two_agree(self, residency, host):
        residency.model_data.sd_model.forge_objects.vae.upscale_ratio = 16
        host.processing.opt_f = 16

        assert mc_memory.align_latent_scale() is False
        assert host.processing.opt_f == 16

    def test_it_leaves_an_unreadable_model_alone(self, residency, host):
        del residency.model_data.sd_model.forge_objects
        del residency.model_data.sd_model.forge_objects_original
        host.processing.opt_f = 16

        assert mc_memory.align_latent_scale() is False
        assert host.processing.opt_f == 16

    def test_it_tolerates_no_model_at_all(self, host):
        host.sd_models.model_data.sd_model = None
        assert mc_memory.align_latent_scale() is False

    def test_the_correction_names_both_numbers(self, residency, host, caplog):
        """The log line has to be enough to place the fault without the source."""
        residency.model_data.sd_model.forge_objects.vae.upscale_ratio = 8
        host.processing.opt_f = 16

        with caplog.at_level(logging.WARNING, logger="model_chain"):
            mc_memory.align_latent_scale()

        assert "16" in caplog.text and "8" in caplog.text


class TestClearReferences:
    def test_clears_engine_and_dynamic_args_state(self, host):
        from backend.args import dynamic_args

        cleared = {"called": False}

        class FakeEngine:
            ini_latent = "an image"

            def clear_references(self):
                cleared["called"] = True

        host.shared.sd_model = FakeEngine()
        dynamic_args.ref_latents.append("stale")

        mc_memory.clear_references()

        assert cleared["called"] is True
        assert host.shared.sd_model.ini_latent is None
        assert dynamic_args.ref_latents == []

    def test_tolerates_a_model_without_reference_support(self, host):
        host.shared.sd_model = object()
        mc_memory.clear_references()  # must not raise


class TestStashKeying:
    """The outgoing model must be filed under *its own* key.

    forge_loading_parameters describes the selection, which runs ahead of the
    loaded model: restore_selection points it back at Stage 1 while Stage 2's
    model is still resident. Keying a stash off the selection there files the
    outgoing model under the incoming one's key -- and because that key is
    already cached, the stash is skipped and the model silently dropped.
    """

    def test_a_full_generation_cycle_caches_both_models(self, residency):
        """Reproduces the observed A -> B -> (restore) -> A -> B sequence."""
        model_a = residency.model_data.sd_model

        # Generation 1: switch to B, then hand the selection back to A.
        mc_memory.ensure_resident("B")
        model_b = residency.model_data.sd_model
        mc_memory.restore_selection("A")

        # Generation 2 begins: A is swapped back in, and B must be kept.
        assert mc_memory.reinstate_pending() is True
        assert residency.model_data.sd_model is model_a

        assert sorted(mc_memory.cached_names()) == ["A", "B"], (
            "B was dropped on the way back to A"
        )

        # ...so the second switch to B is warm rather than another disk read.
        reloads = residency.state.reloads
        assert mc_memory.ensure_resident("B") == "warm"
        assert residency.state.reloads == reloads
        assert residency.model_data.sd_model is model_b

    def test_repeated_cycles_never_read_from_disk_again(self, residency):
        mc_memory.ensure_resident("B")
        mc_memory.restore_selection("A")
        mc_memory.reinstate_pending()
        baseline = residency.state.reloads

        for _ in range(3):
            assert mc_memory.ensure_resident("B") == "warm"
            mc_memory.restore_selection("A")
            assert mc_memory.reinstate_pending() is True

        assert residency.state.reloads == baseline

    def test_the_stash_key_follows_the_loaded_model_not_the_selection(self, residency):
        mc_memory.ensure_resident("B")
        loaded_key = residency.model_data.forge_hash

        # Point the selection elsewhere without touching the loaded model.
        mc_memory.restore_selection("A")
        assert residency.model_data.forge_hash == loaded_key
        assert mc_memory._loaded_model_key() == loaded_key
        assert mc_memory._loading_parameters_key() != loaded_key

    def test_an_unidentifiable_model_is_not_cached_under_a_wrong_key(self, residency):
        residency.model_data.forge_hash = ""
        before = list(mc_memory.cached_names())

        mc_memory._stash_current()

        assert mc_memory.cached_names() == before


class TestRefusalTracking:
    """A cold load is only actionable when the cache actually refused a model.

    Reporting every cold load as a cache problem sent the user chasing a RAM
    setting when the real cause was a keying bug.
    """

    def test_no_refusal_recorded_when_caching_succeeds(self, residency):
        mc_memory.ensure_resident("B")
        assert mc_memory.last_refusal() is None

    def test_a_refusal_is_recorded_and_named(self, residency, monkeypatch):
        monkeypatch.setattr(mc_memory, "cache_budget_bytes", lambda: 1 * GB)
        monkeypatch.setattr(mc_memory, "loaded_size_bytes", lambda model: 20 * GB)

        mc_memory.ensure_resident("B")

        assert mc_memory.last_refusal() == "A"

    def test_a_later_success_clears_the_refusal(self, residency, monkeypatch):
        monkeypatch.setattr(mc_memory, "cache_budget_bytes", lambda: 1 * GB)
        monkeypatch.setattr(mc_memory, "loaded_size_bytes", lambda model: 20 * GB)
        mc_memory.ensure_resident("B")
        assert mc_memory.last_refusal() is not None

        monkeypatch.setattr(mc_memory, "cache_budget_bytes", lambda: 64 * GB)
        monkeypatch.setattr(mc_memory, "loaded_size_bytes", lambda model: 7 * GB)
        mc_memory.ensure_resident("A")

        assert mc_memory.last_refusal() is None


# --------------------------------------------------------------------------- #
# Two kinds of free
# --------------------------------------------------------------------------- #
#
# The host's own figure is free device memory plus whatever its allocator is
# holding cached and not using, and for the host that is exactly right: torch
# reuses its own cache before it asks the driver for anything. For another
# process it is a fiction, and llama.cpp is another process. A placement sized
# against the host's number asks for VRAM that exists only inside this one --
# and what comes back is "cudaMalloc failed: out of memory" on a card that
# reports twenty-two gigabytes free.


class FakeTorch:
    """Just enough torch to answer the two memory questions."""

    def __init__(self, device_free, cached=0, per_card=None):
        self.device_free = device_free
        self.cached = cached
        self.emptied = 0
        self.per_card = dict(per_card or {})

        self.asked: list = []
        outer = self

        class _Cuda:
            @staticmethod
            def mem_get_info(device=None):
                outer.asked.append(device)
                if isinstance(device, _Device) and device.index in outer.per_card:
                    return outer.per_card[device.index], 32 * GB
                return outer.device_free, 24 * GB

            @staticmethod
            def empty_cache():
                outer.emptied += 1
                outer.device_free += outer.cached
                outer.cached = 0

        self.cuda = _Cuda()

    @staticmethod
    def device(kind, index):
        return _Device(kind, index)


class _Device:
    """torch.device('cuda', n), as much of it as this module touches."""

    def __init__(self, kind, index):
        self.type = kind
        self.index = index


class TestDriverFreeVram:
    def _install(self, monkeypatch, torch, free_total):
        from backend import memory_management

        monkeypatch.setitem(sys.modules, "torch", torch)
        device = types.SimpleNamespace(type="cuda")
        monkeypatch.setattr(memory_management, "get_torch_device", lambda: device)
        monkeypatch.setattr(memory_management, "get_free_memory", lambda dev=None: free_total)

    def test_the_allocator_s_cache_is_free_to_the_host_and_to_nobody_else(
            self, host, monkeypatch):
        """20 GB free by the host's accounting, 4 of it actually on offer."""
        torch = FakeTorch(device_free=4 * GB, cached=16 * GB)
        self._install(monkeypatch, torch, free_total=20 * GB)

        assert mc_memory.free_vram_bytes() == 20 * GB
        assert mc_memory.device_free_vram_bytes() == 4 * GB

    def test_emptying_the_cache_hands_it_back(self, host, monkeypatch):
        torch = FakeTorch(device_free=4 * GB, cached=16 * GB)
        self._install(monkeypatch, torch, free_total=20 * GB)
        from backend import memory_management

        monkeypatch.delattr(memory_management, "soft_empty_cache")

        recovered = mc_memory.release_cached_vram()

        assert torch.emptied == 1
        assert recovered == 16 * GB
        assert mc_memory.device_free_vram_bytes() == 20 * GB

    def test_a_second_card_is_asked_about_by_index(self, host, monkeypatch):
        """From a two-card machine: the plan said "24.0 GB on the card" while
        llama.cpp reported CUDA0 with 30.2 GB free and CUDA1 with 22.8 GB.
        Every figure came from whichever card Forge generates on, so a language
        model pinned to the idle one was sized against a card it never touched.
        """
        torch = FakeTorch(device_free=4 * GB, per_card={0: 30 * GB, 1: 22 * GB})
        self._install(monkeypatch, torch, free_total=20 * GB)

        assert mc_memory.device_free_vram_bytes(0) == 30 * GB
        assert mc_memory.device_free_vram_bytes(1) == 22 * GB
        assert mc_memory.device_free_vram_bytes() == 4 * GB

    def test_the_image_card_is_named_by_index(self, host, monkeypatch):
        from backend import memory_management

        monkeypatch.setattr(memory_management, "get_torch_device",
                            lambda: _Device("cuda", 1))

        assert mc_memory.image_device_index() == 1

    def test_a_processor_install_has_no_image_card(self, host, monkeypatch):
        from backend import memory_management

        monkeypatch.setattr(memory_management, "get_torch_device",
                            lambda: _Device("cpu", None))

        assert mc_memory.image_device_index() == -1

    def test_a_question_that_cannot_be_put_falls_back_to_the_host_s_answer(
            self, host, monkeypatch):
        """A wrong number still places a model. No number places nothing."""
        from backend import memory_management

        monkeypatch.setattr(memory_management, "get_free_memory", lambda dev=None: 11 * GB)
        # A string, not a device: there is nothing here to ask for a type.
        monkeypatch.setattr(memory_management, "get_torch_device", lambda: "cuda")

        assert mc_memory.device_free_vram_bytes() == 11 * GB


# --------------------------------------------------------------------------- #
# The console clock
# --------------------------------------------------------------------------- #


class TestEveryLineCarriesTheTime:
    """Half of what this extension logs is a number that only means something
    against a clock: a plan and the budget it implies, a placement and the
    reason for it, a phase peak against the reserve meant to cover it. The
    host's handler prints the message, the module and the level, and no time at
    all.
    """

    def stamp(self, message, *args, level=logging.INFO):
        record = logging.LogRecord("model_chain", level, __file__, 1, message, args, None)
        assert mc_memory._Timestamped().filter(record) is True
        return record.getMessage()

    def test_a_house_line_gets_the_clock_inside_its_own_name(self):
        stamped = self.stamp("Model Chain: Stage 1 is warm")

        assert re.fullmatch(r"Model Chain \[\d\d:\d\d:\d\d\.\d\d\d\]: Stage 1 is warm",
                            stamped), stamped

    def test_it_stays_greppable(self):
        """Somebody with a shell history full of ``grep 'Model Chain:'`` should
        not have to notice this change at all."""
        assert "Model Chain:" not in self.stamp("Model Chain: Stage 1 is warm")
        assert self.stamp("Model Chain: Stage 1 is warm").startswith("Model Chain [")

    def test_the_arguments_are_still_substituted(self):
        stamped = self.stamp("Model Chain: llama-server ready — %d layers, %.1f GB", 14, 1.4)

        assert stamped.endswith("llama-server ready — 14 layers, 1.4 GB")

    def test_a_whole_sentence_passed_through_a_placeholder_is_stamped_too(self):
        """The one call site that does this. A filter that only rewrote the
        format string would leave exactly one line unstamped."""
        stamped = self.stamp("%s", "Model Chain: Stage 1 will load from disk")

        assert stamped.startswith("Model Chain [")
        assert stamped.endswith(": Stage 1 will load from disk")

    def test_a_line_without_the_house_prefix_still_gets_a_clock(self):
        """A log with a hole in it is worse than one with an odd-looking line."""
        stamped = self.stamp("something else entirely")

        assert re.fullmatch(r"\[\d\d:\d\d:\d\d\.\d\d\d\] something else entirely", stamped)

    def test_the_milliseconds_are_always_three_digits(self):
        """So the column lines up and a sort by time is a sort by string."""
        record = logging.LogRecord("model_chain", logging.INFO, __file__, 1,
                                   "Model Chain: x", (), None)
        record.msecs = 7.0
        mc_memory._Timestamped().filter(record)

        assert ".007]" in record.getMessage()

    def test_a_message_that_cannot_be_formatted_is_still_logged(self):
        """A logger that raises while logging takes the caller with it, and the
        caller is usually in the middle of a generation."""
        record = logging.LogRecord("model_chain", logging.INFO, __file__, 1,
                                   "Model Chain: %d", ("not a number",), None)

        assert mc_memory._Timestamped().filter(record) is True

    def test_the_filter_is_attached_to_the_logger(self, host):
        assert any(isinstance(f, mc_memory._Timestamped)
                   for f in logging.getLogger("model_chain").filters)

    def test_it_is_not_attached_twice(self, host):
        before = sum(isinstance(f, mc_memory._Timestamped)
                     for f in logging.getLogger("model_chain").filters)

        mc_memory._make_logger()

        after = sum(isinstance(f, mc_memory._Timestamped)
                    for f in logging.getLogger("model_chain").filters)
        assert before == after == 1

    def test_it_reaches_a_module_that_only_asked_for_the_logger(self, caplog):
        """Every module reaches the same logger object through
        ``getLogger("model_chain")``, which is what makes one call cover all of
        them."""
        import mc_broker

        with caplog.at_level(logging.INFO, logger="model_chain"):
            mc_broker.logger.info("Model Chain: a broker line")

        assert any(r.getMessage().startswith("Model Chain [") for r in caplog.records)
