"""Prepared model + LoRA state: reuse it when safe, and never leak it sideways.

Two halves, and they fail in opposite directions, which is why both are tested
hard:

* reusing prepared state that is no longer valid produces a *wrong image* with
  no error anywhere -- stale LoRA, stale weight, a LoRA from the other stage;
* refusing to reuse valid prepared state only costs time.

So every test here that asserts preservation also has a sibling asserting the
invalidation, and the invalidation cases outnumber the preservation ones.
"""

from __future__ import annotations

import types

import pytest

import mc_lora
import mc_memory

GB = 1024**3


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #


def make_model(name="A", lora_hash=None):
    """A cached ``sd_model``, optionally carrying a prepared LoRA state."""
    model = types.SimpleNamespace(
        sd_checkpoint_info=types.SimpleNamespace(
            filename=f"/models/{name}", name_for_extra=name, sha256=f"sha-{name}"
        ),
        forge_objects=types.SimpleNamespace(unet=None, clip=None, vae=None),
    )
    # Set unconditionally: the host defines the attribute on every sd_model, and
    # its *value* is what says whether a LoRA is applied.
    model.current_lora_hash = lora_hash or ""
    return model


@pytest.fixture
def cache(host, monkeypatch):
    """A reset cache, with the host's key lookups pointed at a scratch value."""
    mc_memory._cache.clear()
    mc_memory._pending_restore = None

    state = types.SimpleNamespace(key="key-A", loaded="key-A")
    monkeypatch.setattr(mc_memory, "_loading_parameters_key", lambda: state.key)
    monkeypatch.setattr(mc_memory, "_loaded_model_key", lambda: state.loaded)
    monkeypatch.setattr(mc_memory, "loaded_size_bytes", lambda model: 8 * GB)
    monkeypatch.setattr(mc_memory, "file_size_bytes", lambda name, mods=None: 8 * GB)
    monkeypatch.setattr(mc_memory, "free_ram_bytes", lambda: 64 * GB)
    monkeypatch.setattr(mc_memory, "total_ram_bytes", lambda: 96 * GB)

    yield state

    mc_memory._cache.clear()
    mc_memory._pending_restore = None


def stash(host, cache, model, key, stage):
    """Put ``model`` into the cache the way a real switch would."""
    host.sd_models.model_data.sd_model = model
    cache.loaded = key
    mc_memory._stash_current(stage=stage)
    return mc_memory._cache.get(key)


# --------------------------------------------------------------------------- #
# Keeping a LoRA in its own stage
# --------------------------------------------------------------------------- #


class TestStageIsolation:
    """``all_prompts`` keeps the tags the text encoder never sees.

    The host strips extra-network tags from a *copy* of the prompt on its way to
    the encoder -- which is why they still show up in infotext -- so a Stage 2
    pass handed ``processed.all_prompts[i]`` verbatim would parse and apply
    Stage 1's LoRAs against Model B.
    """

    @pytest.mark.parametrize(
        "prompt, expected",
        [
            ("a castle <lora:detail:0.8>", "a castle"),
            ("<lora:detail:0.8> a castle", "a castle"),
            ("a castle, <lora:detail:0.8>, at dusk", "a castle, at dusk"),
            ("a castle <lyco:style:1>", "a castle"),
            ("a castle <hypernet:thing:0.5>", "a castle"),
            ("<lora:a:1><lora:b:1>", ""),
            ("a castle", "a castle"),
            ("", ""),
        ],
    )
    def test_tags_are_removed_and_the_gap_closed(self, prompt, expected):
        assert mc_lora.strip_networks(prompt)[0] == expected

    def test_the_removed_tags_are_reported(self):
        _, removed = mc_lora.strip_networks("a castle <lora:detail:0.8> <lora:film:0.4>")
        assert removed == ["<lora:detail:0.8>", "<lora:film:0.4>"]

    def test_a_prompt_without_tags_is_returned_unchanged(self):
        """Not merely equal -- untouched. Tidying is only ever a repair."""
        prompt = "a castle,  with  odd   spacing, "
        assert mc_lora.strip_networks(prompt) == (prompt, [])

    def test_prompt_scheduling_is_not_a_network_tag(self):
        """``[a:b:0.5]`` uses brackets; only ``<...>`` is an extra network."""
        prompt = "a [castle:tower:0.5] at dusk"
        assert mc_lora.strip_networks(prompt) == (prompt, [])


class TestStageIsolationInTheChain:
    def test_inherit_does_not_carry_a_stage_1_lora(self, chain):
        positive, _ = chain.script._resolve_prompts(
            "a castle <lora:sdxl_detail:0.8>", "lowres", "Inherit", "", "", []
        )
        assert "<lora:" not in positive

    def test_append_does_not_carry_a_stage_1_lora_either(self, chain):
        positive, _ = chain.script._resolve_prompts(
            "a castle <lora:sdxl_detail:0.8>", "lowres", "Append", "cinematic", "", []
        )
        assert positive == "a castle, cinematic"

    def test_a_stage_2_lora_survives_append(self, chain):
        """The Stage 2 boxes are how a LoRA is *meant* to be asked for."""
        positive, _ = chain.script._resolve_prompts(
            "a castle <lora:sdxl_detail:0.8>", "", "Append", "<lora:flux_edit:1.0>", "", []
        )
        assert positive == "a castle, <lora:flux_edit:1.0>"

    def test_a_stage_2_lora_survives_replace(self, chain):
        positive, _ = chain.script._resolve_prompts(
            "a castle <lora:sdxl_detail:0.8>", "", "Replace", "<lora:flux_edit:1.0>", "", []
        )
        assert positive == "<lora:flux_edit:1.0>"

    def test_negative_prompts_are_stripped_too(self, chain):
        _, negative = chain.script._resolve_prompts(
            "a castle", "lowres <lora:bad:0.5>", "Inherit", "", "", []
        )
        assert "<lora:" not in negative

    def test_the_refine_pass_receives_the_stripped_prompt(self, chain, host, image_factory):
        """End to end: what actually reaches Model B carries no Stage 1 LoRA."""
        from test_orchestration import make_p, make_processed, run_chain

        p = make_p(host, prompt="a castle <lora:sdxl_detail:0.8>")
        run_chain(chain, host, p, make_processed(host, p, image_factory), prompt_mode="Inherit")

        assert chain.refine_calls
        assert "<lora:" not in chain.refine_calls[0].prompt

    def test_a_generation_without_tags_is_unaffected(self, chain, host, image_factory):
        from test_orchestration import make_p, make_processed, run_chain

        p = make_p(host, prompt="a castle")
        run_chain(chain, host, p, make_processed(host, p, image_factory))

        assert chain.refine_calls[0].prompt == "a castle"


# --------------------------------------------------------------------------- #
# Reusing a prepared state
# --------------------------------------------------------------------------- #


class TestPreparedStateIsReused:
    def test_an_unchanged_state_survives_the_round_trip(self, host, cache):
        """The point of the feature: no reapplication for an unchanged LoRA.

        Preservation here means *not* invalidating -- the host's own hash
        comparison then returns early on its next call, which is the mechanism
        that skips the work.
        """
        model = make_model("A", lora_hash="str(['detail'], [0.8], [0.8], [None])")
        entry = stash(host, cache, model, "key-A", mc_memory.STAGE_1)

        assert mc_memory._restore_prepared_state(entry, mc_memory.STAGE_1) == "preserved"
        assert model.current_lora_hash == "str(['detail'], [0.8], [0.8], [None])"

    def test_a_model_with_no_lora_reports_nothing_either_way(self, host, cache):
        model = make_model("A")
        entry = stash(host, cache, model, "key-A", mc_memory.STAGE_1)

        assert mc_memory._restore_prepared_state(entry, mc_memory.STAGE_1) == "none"

    def test_repeated_round_trips_do_not_drift(self, host, cache):
        """Soak: twenty alternating swaps must not accumulate anything."""
        model = make_model("A", lora_hash="hash-1")

        for _ in range(20):
            entry = stash(host, cache, model, "key-A", mc_memory.STAGE_1)
            assert mc_memory._restore_prepared_state(entry, mc_memory.STAGE_1) == "preserved"

        assert model.current_lora_hash == "hash-1"
        assert len(mc_memory.cached_names()) == 1


class TestPreparedStateIsInvalidated:
    """Each of these is a way for a preserved state to become the wrong state."""

    def test_a_state_that_changed_while_cached_is_rebuilt(self, host, cache):
        model = make_model("A", lora_hash="hash-1")
        entry = stash(host, cache, model, "key-A", mc_memory.STAGE_1)

        model.current_lora_hash = "hash-2"  # something moved it behind our back

        assert mc_memory._restore_prepared_state(entry, mc_memory.STAGE_1) == "rebuilt"
        assert model.current_lora_hash == mc_lora.REBUILD

    def test_the_other_stage_never_inherits_a_prepared_state(self, host, cache):
        """One checkpoint used by both stages is the cross-stage leak case."""
        model = make_model("A", lora_hash="hash-1")
        entry = stash(host, cache, model, "key-A", mc_memory.STAGE_1)

        assert mc_memory._restore_prepared_state(entry, mc_memory.STAGE_2) == "rebuilt"

    def test_a_backend_that_rebuilds_its_own_state_is_not_preserved(self, host, cache):
        """Nunchaku folds LoRA into a quantised kernel, not a patcher clone."""
        from backend.args import dynamic_args

        dynamic_args.nunchaku = True
        model = make_model("A", lora_hash="hash-1")
        entry = stash(host, cache, model, "key-A", mc_memory.STAGE_1)

        assert entry.lora_preservable is False
        assert mc_memory._restore_prepared_state(entry, mc_memory.STAGE_1) == "rebuilt"

    def test_the_setting_turns_preservation_off(self, host, cache):
        host.shared.opts.model_chain_preserve_lora = False
        model = make_model("A", lora_hash="hash-1")
        entry = stash(host, cache, model, "key-A", mc_memory.STAGE_1)

        assert mc_memory._restore_prepared_state(entry, mc_memory.STAGE_1) == "rebuilt"

    def test_preservation_is_on_when_the_setting_is_absent(self, host, cache, monkeypatch):
        """A host that never registered the option still gets the default.

        Which is the case for a config saved by a version that predates it, and
        for any code path that reads the option before the settings page has
        been built.
        """
        monkeypatch.delitem(host.shared.options_templates, mc_memory.OPT_PRESERVE_LORA)
        assert not hasattr(host.shared.opts, mc_memory.OPT_PRESERVE_LORA)

        model = make_model("A", lora_hash="hash-1")
        entry = stash(host, cache, model, "key-A", mc_memory.STAGE_1)

        assert mc_memory._restore_prepared_state(entry, mc_memory.STAGE_1) == "preserved"

    def test_restashing_records_the_state_being_put_away_now(self, host, cache):
        """A cached entry describes this visit, not the first one.

        Otherwise a model that came back, had a different LoRA applied and went
        away again would be compared against a hash two jobs old.
        """
        model = make_model("A", lora_hash="hash-1")
        stash(host, cache, model, "key-A", mc_memory.STAGE_1)

        model.current_lora_hash = "hash-2"
        entry = stash(host, cache, model, "key-A", mc_memory.STAGE_1)

        assert entry.lora_state == "hash-2"
        assert mc_memory._restore_prepared_state(entry, mc_memory.STAGE_1) == "preserved"


class TestFailureDoesNotPoisonTheCache:
    """A LoRA that raised half way through leaves a belief that outlives the job.

    Model Chain keeps the model object rather than reloading it, so unlike a
    stock Forge session that belief is not cleared by the next checkpoint load.
    """

    def test_a_failed_refine_invalidates_the_prepared_state(self, chain, host, image_factory, monkeypatch):
        from test_orchestration import make_p, make_processed, run_chain

        model = make_model("B", lora_hash="hash-1")
        host.sd_models.model_data.sd_model = model

        def explode(p2):
            raise RuntimeError("LoRA is incompatible with this architecture")

        monkeypatch.setattr(chain.module, "process_images", explode)

        p = make_p(host)
        run_chain(chain, host, p, make_processed(host, p, image_factory))

        assert model.current_lora_hash == mc_lora.REBUILD

    def test_a_successful_refine_leaves_the_state_alone(self, chain, host, image_factory):
        from test_orchestration import make_p, make_processed, run_chain

        model = make_model("B", lora_hash="hash-1")
        host.sd_models.model_data.sd_model = model

        p = make_p(host)
        run_chain(chain, host, p, make_processed(host, p, image_factory))

        assert model.current_lora_hash == "hash-1"


class TestInvalidate:
    def test_it_writes_a_value_no_real_hash_can_match(self, host):
        model = make_model("A", lora_hash="hash-1")
        assert mc_lora.invalidate(model, "testing") is True
        assert mc_lora.state_of(model) is None

    def test_a_model_the_host_never_touched_has_nothing_to_invalidate(self, host):
        bare = types.SimpleNamespace()
        assert mc_lora.invalidate(bare, "testing") is False

    def test_it_survives_a_model_that_refuses_the_write(self, host):
        class Frozen:
            current_lora_hash = "hash-1"

            def __setattr__(self, name, value):
                raise AttributeError("read-only")

        assert mc_lora.invalidate(Frozen(), "testing") is False

    def test_a_module_level_host_hash_is_cleared_too(self, host, monkeypatch):
        """A stale global would describe a model that is no longer loaded."""
        import sys

        networks = types.ModuleType("networks")
        networks.current_lora_hash = "hash-1"
        monkeypatch.setitem(sys.modules, "networks", networks)

        mc_lora.invalidate(make_model("A", lora_hash="hash-1"), "testing")

        assert networks.current_lora_hash == mc_lora.REBUILD

    def test_a_host_without_that_global_is_left_alone(self, host, monkeypatch):
        import sys

        networks = types.ModuleType("networks")
        monkeypatch.setitem(sys.modules, "networks", networks)

        mc_lora.invalidate(make_model("A", lora_hash="hash-1"), "testing")

        assert not hasattr(networks, "current_lora_hash")


class TestPreservability:
    def test_an_ordinary_model_is_preservable(self):
        assert mc_lora.is_preservable({"nunchaku": False})[0] is True

    def test_nunchaku_is_not(self):
        preservable, reason = mc_lora.is_preservable({"nunchaku": True})
        assert preservable is False
        assert "nunchaku" in reason

    def test_missing_flags_are_treated_as_an_ordinary_model(self):
        assert mc_lora.is_preservable(None)[0] is True
        assert mc_lora.is_preservable({})[0] is True
