"""Warm LoRA, Cold LoRA: the un-merged originals, and the blob that has none.

The two ways this can go wrong are not symmetric. Freeing originals that the
host still needs costs nothing visible until the next LoRA change, when the
host would merge the new set on top of the old one and every image after that
is quietly wrong. Keeping originals that could have gone costs twelve gigabytes
of RAM. So the tests of the rule *nothing is ever merged onto a blob* outnumber
the tests of the freeing, and the freeing is checked to happen only when the
host's own stamp says the merge is complete.
"""

from __future__ import annotations

import types

import pytest

import mc_lora
import mc_lora_ram
import mc_memory

GB = 1024 ** 3
MB = 1024 ** 2


# --------------------------------------------------------------------------- #
# Fakes: the patcher as Forge Neo's ModelPatcher keeps its LoRA state
# --------------------------------------------------------------------------- #


class FakeTensor:
    def __init__(self, count):
        self.count = count

    def numel(self):
        return self.count

    def element_size(self):
        return 2


class KModel:
    """The inner module: carries the host's merge stamp and the lowvram flags."""

    def __init__(self, stamp):
        self.current_weight_patches_uuid = stamp
        self.model_lowvram = False
        self.lowvram_patch_counter = 0


class FakePatcher:
    """A patcher clone the way ``load_lora_for_models`` leaves one.

    ``patches`` holds the LoRA, ``backup`` the originals of the same keys (and,
    for one key, an original the host filed for a reason of its own, which a
    bake must leave alone), and the inner model's stamp says whether ``load``
    has merged them.
    """

    def __init__(self, keys=("a", "b"), merged=True, mb_each=512, online=False):
        self.patches = {key: ["lora"] for key in keys}
        self.patches_uuid = "uuid-merged"
        self.model = KModel("uuid-merged" if merged else "uuid-before")
        self.backup = {key: FakeTensor(mb_each * MB // 2) for key in keys}
        self.backup["cast.weight"] = FakeTensor(MB // 2)
        self._online = online

    def has_online_lora(self):
        return self._online

    def model_size(self):
        return 12 * GB


def make_model(lora_hash="str([['/loras/x.safetensors', 1.0, 1.0, False]])", name="A", **patcher):
    unet = FakePatcher(**patcher)
    clip = types.SimpleNamespace(patcher=FakePatcher(keys=("te",), mb_each=64, **patcher))
    objects = types.SimpleNamespace(unet=unet, clip=clip, vae=None)
    model = types.SimpleNamespace(
        sd_checkpoint_info=types.SimpleNamespace(
            filename=f"/models/{name}", name_for_extra=name, sha256=f"sha-{name}"),
        forge_objects=objects,
        # As the host leaves them: the loader's clone kept here, and the
        # original patchers kept apart.
        forge_objects_after_applying_lora=types.SimpleNamespace(unet=unet, clip=clip, vae=None),
        forge_objects_original=types.SimpleNamespace(
            unet=types.SimpleNamespace(patches={}, patches_uuid="uuid-base", backup=unet.backup,
                                       model=unet.model),
            clip=clip, vae=None),
        current_lora_hash=lora_hash,
    )
    return model


def make_p(prompt="a castle <lora:x:1.0>", negative="", styles=None, **extra):
    p = types.SimpleNamespace(prompt=prompt, negative_prompt=negative, styles=styles or [],
                              enable_hr=False, comments=[])
    p.comment = p.comments.append
    for key, value in extra.items():
        setattr(p, key, value)
    return p


def originals(model):
    """The LoRA keys' originals still held, per patcher."""
    return [sorted(key for key in patcher.backup if key != "cast.weight")
            for patcher in mc_memory.model_patchers(model)]


@pytest.fixture
def loaded(host, monkeypatch):
    """A merged LoRA'd model in the host's slot, mode Warm, nothing pending."""
    mc_lora_ram.reset_for_tests()
    mc_memory._cache.clear()
    mc_memory._pending_restore = None
    host.shared.opts.model_chain_lora_ram = mc_lora_ram.WARM
    host.shared.opts.sd_lora = "None"
    host.sd_models.unloaded.clear()
    model = make_model()
    model_data = host.sd_models.model_data
    model_data.sd_model = model
    model_data.forge_hash = "key-A"
    model_data.set_sd_model = lambda v: setattr(model_data, "sd_model", v)
    monkeypatch.setattr(mc_memory, "free_ram_bytes", lambda: 40 * GB)
    yield model
    mc_lora_ram.reset_for_tests()
    mc_memory._cache.clear()
    mc_memory._pending_restore = None


def dropped(host):
    """Whether the loaded model went through the host's own Unload."""
    return bool(host.sd_models.unloaded)


# --------------------------------------------------------------------------- #
# The mode
# --------------------------------------------------------------------------- #


class TestTheMode:
    def test_warm_is_the_default_and_anything_unreadable_is_warm(self, host):
        assert mc_lora_ram.mode() == mc_lora_ram.WARM
        host.shared.opts.model_chain_lora_ram = "Lukewarm"
        assert mc_lora_ram.mode() == mc_lora_ram.WARM
        host.shared.opts.model_chain_lora_ram = "cold"
        assert mc_lora_ram.mode() == mc_lora_ram.COLD

    def test_the_api_words_and_the_labels_are_the_same_two_modes(self):
        assert mc_lora_ram.normalise("cold") == mc_lora_ram.COLD
        assert mc_lora_ram.normalise("Warm") == mc_lora_ram.WARM
        assert mc_lora_ram.normalise(" COLD ") == mc_lora_ram.COLD
        assert mc_lora_ram.normalise("hot") is None
        with pytest.raises(ValueError):
            mc_lora_ram.set_mode("hot")

    def test_the_mode_is_stored_and_saved_the_way_settings_stores_it(self, loaded, host):
        found = mc_lora_ram.set_mode("cold")

        assert host.shared.opts.model_chain_lora_ram == mc_lora_ram.COLD
        assert host.shared.opts.data[mc_lora_ram.OPT_LORA_RAM] == mc_lora_ram.COLD
        assert found["mode"] == "cold"


# --------------------------------------------------------------------------- #
# Cold frees the originals -- at once, and only a finished merge
# --------------------------------------------------------------------------- #


class TestColdFreesTheOriginals:
    def test_choosing_cold_frees_the_originals_at_once_and_bakes_the_model(self, loaded, host):
        """"The setting should immediately free up the ram.\""""
        assert originals(loaded) == [["a", "b"], ["te"]]

        found = mc_lora_ram.set_mode("cold")

        assert originals(loaded) == [[], []]
        assert found["blob"] is True and found["merged"] is False
        assert found["originals_bytes"] == 0
        assert found["message"].startswith("Cold LoRA: 1.1 GB of un-merged weights freed")
        blob = mc_lora_ram.blob_of(loaded)
        assert blob.hash == loaded.current_lora_hash
        assert blob.freed_bytes == (512 + 512 + 64) * MB

    def test_the_patches_go_with_the_originals_so_the_host_cannot_merge_again(self, loaded):
        """With the originals gone the host would merge onto merged weights on
        its next ``load``; with no patches left it has nothing to merge."""
        mc_lora_ram.set_mode("cold")

        for patcher in mc_memory.model_patchers(loaded):
            assert patcher.patches == {}
            assert patcher.patches_uuid == patcher.model.current_weight_patches_uuid

    def test_the_loaders_clone_is_baked_even_when_another_extension_left_its_own_in_front(self, loaded):
        """``forge_objects`` is reset from ``forge_objects_after_applying_lora``
        at every batch, so the LoRA lives in the loader's clone whatever a
        generation left in ``forge_objects``. A bake that missed it would hand
        the next generation a clone with patches over merged weights."""
        loaders = loaded.forge_objects_after_applying_lora.unet
        transient = FakePatcher()                  # another extension's clone, patches copied
        transient.backup = loaders.backup
        transient.model = loaders.model
        loaded.forge_objects.unet = transient

        mc_lora_ram.set_mode("cold")

        assert loaders.patches == {} and transient.patches == {}
        assert [key for key in loaders.backup if key != "cast.weight"] == []

    def test_an_original_the_host_filed_for_its_own_reason_is_left_alone(self, loaded):
        mc_lora_ram.set_mode("cold")

        for patcher in mc_memory.model_patchers(loaded):
            assert "cast.weight" in patcher.backup

    def test_the_hosts_record_of_the_merged_set_is_kept(self, loaded):
        before = loaded.current_lora_hash
        mc_lora_ram.set_mode("cold")
        assert loaded.current_lora_hash == before, "the host's early return still fires"

    def test_a_merge_the_host_has_not_finished_is_not_baked(self, loaded, host):
        host.sd_models.model_data.sd_model = model = make_model(merged=False)

        found = mc_lora_ram.set_mode("cold")

        assert originals(model) == [["a", "b"], ["te"]]
        assert mc_lora_ram.blob_of(model) is None
        assert "nothing freed" in found["message"] and "not finished merging" in found["message"]

    def test_a_model_only_partly_on_the_card_is_not_baked(self, loaded):
        """The host applies the LoRA to offloaded layers at run time from the
        very patches a bake would clear."""
        loaded.forge_objects.unet.model.model_lowvram = True

        found = mc_lora_ram.set_mode("cold")

        assert mc_lora_ram.blob_of(loaded) is None
        assert originals(loaded) == [["a", "b"], ["te"]]
        assert "partly on the card" in found["message"]

    def test_a_lora_applied_on_the_fly_is_not_baked(self, loaded, host):
        host.sd_models.model_data.sd_model = model = make_model(online=True)

        found = mc_lora_ram.set_mode("cold")

        assert mc_lora_ram.blob_of(model) is None
        assert "on the fly" in found["message"]

    def test_a_model_with_no_lora_has_nothing_to_free(self, loaded, host):
        host.sd_models.model_data.sd_model = model = make_model(lora_hash=mc_lora.NO_NETWORKS)

        found = mc_lora_ram.set_mode("cold")

        assert mc_lora_ram.blob_of(model) is None
        assert "nothing to free" in found["message"]

    def test_nothing_loaded_is_not_an_error(self, loaded, host):
        host.sd_models.model_data.sd_model = host.sd_models.FakeInitialModel()

        found = mc_lora_ram.set_mode("cold")

        assert found["mode"] == "cold" and found["blob"] is False

    def test_cold_chosen_during_a_generation_waits_for_its_end(self, loaded, host):
        host.shared.state.job = "txt2img"

        found = mc_lora_ram.set_mode("cold")

        assert originals(loaded) == [["a", "b"], ["te"]], "never under a running job"
        assert found["pending"] is True
        assert "generation is running" in found["message"]

        host.shared.state.job = ""
        assert mc_lora_ram.after_generation(make_p()) == "baked"
        assert originals(loaded) == [[], []]
        assert mc_lora_ram.status()["pending"] is False

    def test_cold_chosen_while_the_hosts_queue_lock_is_held_waits_too(self, loaded, host):
        """Every GPU call the host makes runs under ``queue_lock``; a held lock
        is a generation whatever ``state.job`` says."""
        lock = host.call_queue.queue_lock
        lock.acquire()
        try:
            found = mc_lora_ram.set_mode("cold")
        finally:
            lock.release()

        assert found["pending"] is True
        assert originals(loaded) == [["a", "b"], ["te"]]
        # And the lock was not left taken.
        assert lock.acquire(blocking=False)
        lock.release()

    def test_the_lock_is_given_back_after_a_bake(self, loaded, host):
        mc_lora_ram.set_mode("cold")
        lock = host.call_queue.queue_lock
        assert lock.acquire(blocking=False)
        lock.release()


# --------------------------------------------------------------------------- #
# Warm back on restores nothing
# --------------------------------------------------------------------------- #


class TestWarmAfterCold:
    def test_warm_does_not_bring_the_originals_back(self, loaded):
        """"Once gone, turning warm back on does not load the lora back!\""""
        mc_lora_ram.set_mode("cold")
        found = mc_lora_ram.set_mode("warm")

        assert originals(loaded) == [[], []]
        assert mc_lora_ram.blob_of(loaded) is not None
        assert found["mode"] == "warm" and found["blob"] is True
        assert "stays baked until its LoRA changes" in found["message"]

    def test_warm_clears_a_pending_bake(self, loaded, host):
        host.shared.state.job = "txt2img"
        mc_lora_ram.set_mode("cold")
        assert mc_lora_ram.status()["pending"] is True

        mc_lora_ram.set_mode("warm")
        host.shared.state.job = ""

        assert mc_lora_ram.status()["pending"] is False
        assert mc_lora_ram.after_generation(make_p()) == "warm"
        assert originals(loaded) == [["a", "b"], ["te"]]

    def test_the_next_merge_stays_warm(self, loaded, host):
        """A blob evicted for a LoRA change reloads from disk; the model that
        comes back is an ordinary model, and Warm leaves its originals alone."""
        mc_lora_ram.set_mode("cold")
        mc_lora_ram.set_mode("warm")
        mc_lora_ram.before_pass(make_p("a castle <lora:x:0.8>"))   # the change evicts the blob
        assert dropped(host)

        # The host reloaded and merged the new set into fresh weights.
        host.sd_models.model_data.sd_model = fresh = make_model(lora_hash="hash-0.8")
        host.sd_models.model_data.forge_hash = "key-A"

        assert mc_lora_ram.after_generation(make_p("a castle <lora:x:0.8>")) == "warm"
        assert originals(fresh) == [["a", "b"], ["te"]]
        assert mc_lora_ram.blob_of(fresh) is None


# --------------------------------------------------------------------------- #
# After a generation
# --------------------------------------------------------------------------- #


class TestAfterAGeneration:
    def test_cold_bakes_what_the_host_merged_against_the_prompt_that_asked(self, loaded, host):
        host.shared.opts.model_chain_lora_ram = mc_lora_ram.COLD

        assert mc_lora_ram.after_generation(make_p("a castle <lora:x:1.0>")) == "baked"
        blob = mc_lora_ram.blob_of(loaded)
        assert blob.composite == mc_lora.composite("<lora:x:1.0>")
        assert originals(loaded) == [[], []]

    def test_warm_leaves_the_merge_alone(self, loaded):
        assert mc_lora_ram.after_generation(make_p()) == "warm"
        assert originals(loaded) == [["a", "b"], ["te"]]

    def test_a_second_pass_on_a_blob_is_nothing_to_do(self, loaded, host):
        host.shared.opts.model_chain_lora_ram = mc_lora_ram.COLD
        assert mc_lora_ram.after_generation(make_p()) == "baked"
        assert mc_lora_ram.after_generation(make_p()) == "blob"

    def test_stage_2s_merge_is_baked_against_the_stage_2_prompts(self, loaded, host):
        host.shared.opts.model_chain_lora_ram = mc_lora_ram.COLD

        assert mc_lora_ram.after_generation(texts=["refined <lora:detail:0.5>"]) == "baked"
        assert mc_lora_ram.blob_of(loaded).composite == mc_lora.composite("<lora:detail:0.5>")

    def test_a_hires_pass_bakes_the_set_it_ended_on(self, loaded, host):
        host.shared.opts.model_chain_lora_ram = mc_lora_ram.COLD
        p = make_p("a castle <lora:x:1.0>", enable_hr=True, hr_prompt="a castle <lora:y:1.0>")

        mc_lora_ram.after_generation(p)

        assert mc_lora_ram.blob_of(loaded).composite == mc_lora.composite("<lora:y:1.0>")


# --------------------------------------------------------------------------- #
# Nothing is ever merged onto a blob: the check before the host's LoRA pass
# --------------------------------------------------------------------------- #


class TestTheBlobIsEvictedBeforeAChange:
    @pytest.fixture
    def blob(self, loaded, host):
        host.shared.opts.model_chain_lora_ram = mc_lora_ram.COLD
        assert mc_lora_ram.after_generation(make_p("a castle <lora:x:1.0>")) == "baked"
        host.sd_models.unloaded.clear()
        return loaded

    def test_the_same_set_runs_on_the_blob_as_it_is(self, blob, host):
        assert mc_lora_ram.before_pass(make_p("a different castle <lora:x:1.0>")) == ""
        assert not dropped(host)
        assert host.sd_models.model_data.sd_model is blob

    def test_case_and_spacing_are_the_same_request(self, blob, host):
        assert mc_lora_ram.before_pass(make_p("x <LoRA:x:1.0>  y")) == ""
        assert not dropped(host)

    def test_a_changed_weight_evicts_the_blob(self, blob, host):
        """"...and evicts it on change... never stack another lora on a blob model!\""""
        reason = mc_lora_ram.before_pass(make_p("a castle <lora:x:0.8>"))

        assert dropped(host)
        assert "<lora:x:0.8>" in reason and "<lora:x:1.0>" in reason
        model_data = host.sd_models.model_data
        assert type(model_data.sd_model).__name__ == "FakeInitialModel"
        assert model_data.forge_hash == "", "the host reloads from disk at this generation"

    def test_a_second_lora_evicts_the_blob(self, blob, host):
        assert mc_lora_ram.before_pass(make_p("a castle <lora:x:1.0> <lora:y:1.0>"))
        assert dropped(host)

    def test_removing_the_lora_evicts_the_blob(self, blob, host):
        """A blob cannot be un-merged: no LoRA is a change like any other."""
        reason = mc_lora_ram.before_pass(make_p("a castle"))

        assert dropped(host)
        assert "no LoRA" in reason

    def test_a_hires_pass_on_another_set_evicts_the_blob_even_when_the_first_pass_matches(
            self, blob, host):
        """The host merges the hires set *during* the generation; a blob under
        it would be merged onto half way through."""
        p = make_p("a castle <lora:x:1.0>", enable_hr=True, hr_prompt="a castle <lora:y:1.0>")

        reason = mc_lora_ram.before_pass(p)

        assert dropped(host)
        assert reason.startswith("the hires pass asks for")

    def test_an_empty_hires_prompt_is_the_first_prompt_again(self, blob, host):
        p = make_p("a castle <lora:x:1.0>", enable_hr=True, hr_prompt="")

        assert mc_lora_ram.before_pass(p) == ""
        assert not dropped(host)

    def test_hires_off_does_not_read_the_hires_prompt(self, blob, host):
        p = make_p("a castle <lora:x:1.0>", enable_hr=False, hr_prompt="a castle <lora:y:1.0>")

        assert mc_lora_ram.before_pass(p) == ""

    def test_a_lora_inside_a_style_counts(self, blob, host, style_store):
        """``setup_prompts`` applies styles before the host parses networks."""
        assert mc_lora_ram.before_pass(make_p("a castle <lora:x:1.0>", styles=["Detailed"])) == ""
        assert not dropped(host)

        assert mc_lora_ram.before_pass(make_p("a castle <lora:x:1.0>", styles=["WithLora"]))
        assert dropped(host)

    def test_the_additional_network_setting_counts(self, blob, host):
        """Settings' *Add network to prompt* reaches every prompt the way a
        typed tag does."""
        host.shared.opts.sd_lora = "always"
        host.shared.opts.extra_networks_default_multiplier = 1.0

        assert mc_lora_ram.before_pass(make_p("a castle <lora:x:1.0>"))
        assert dropped(host)

    def test_the_additional_network_already_in_the_prompt_is_not_counted_twice(self, loaded, host):
        host.shared.opts.sd_lora = "x"
        host.shared.opts.extra_networks_default_multiplier = 1.0
        host.shared.opts.model_chain_lora_ram = mc_lora_ram.COLD
        mc_lora_ram.after_generation(make_p("a castle <lora:x:1.0>"))
        host.sd_models.unloaded.clear()

        assert mc_lora_ram.before_pass(make_p("a castle <lora:x:1.0>")) == ""
        assert not dropped(host)

    def test_a_batch_of_prompts_is_read_whole(self, blob, host):
        assert mc_lora_ram.before_pass(make_p(["a <lora:x:1.0>", "b <lora:x:1.0>"])) == ""
        assert mc_lora_ram.before_pass(make_p(["a <lora:x:1.0>", "b <lora:y:1.0>"]))
        assert dropped(host)

    def test_a_model_with_its_originals_is_never_dropped_whatever_the_prompt(self, loaded, host):
        """Warm, or Cold before any bake: the host un-merges from its backup as
        it always did, and this module has no say."""
        for mode in (mc_lora_ram.WARM, mc_lora_ram.COLD):
            host.shared.opts.model_chain_lora_ram = mode
            assert mc_lora_ram.before_pass(make_p("a castle <lora:y:0.3>")) == ""
            assert not dropped(host)
            assert originals(loaded) == [["a", "b"], ["te"]]

    def test_the_check_runs_whatever_the_mode_is_now(self, blob, host):
        """Warm turned back on does not un-bake the model."""
        host.shared.opts.model_chain_lora_ram = mc_lora_ram.WARM

        assert mc_lora_ram.before_pass(make_p("a castle <lora:x:0.8>"))
        assert dropped(host)

    def test_a_pending_bake_is_made_before_the_comparison(self, loaded, host):
        """Cold chosen mid-generation, then a new prompt: the model is baked
        against this prompt and then compared to it, so a same-set prompt runs
        on it and a changed one evicts it."""
        host.shared.state.job = "txt2img"
        mc_lora_ram.set_mode("cold")
        host.shared.state.job = ""

        assert mc_lora_ram.before_pass(make_p("a castle <lora:x:1.0>")) == ""
        assert mc_lora_ram.blob_of(loaded) is not None and not dropped(host)
        assert mc_lora_ram.status()["pending"] is False

    def test_a_blob_dropped_leaves_nothing_of_it_behind(self, blob, host):
        mc_lora_ram.before_pass(make_p("a castle"))
        assert mc_lora_ram.status()["blob"] is False
        assert mc_lora_ram.before_pass(make_p("a castle")) == ""


# --------------------------------------------------------------------------- #
# The net before sampling
# --------------------------------------------------------------------------- #


class TestTheNetBeforeSampling:
    @pytest.fixture
    def blob(self, loaded, host):
        host.shared.opts.model_chain_lora_ram = mc_lora_ram.COLD
        mc_lora_ram.after_generation(make_p("a castle <lora:x:1.0>"))
        host.sd_models.unloaded.clear()
        return loaded

    def rebuilt_by_the_host(self, model):
        """What ``load_networks`` leaves when it did not take its early return:
        a new hash, and clones with patches and a fresh ``patches_uuid``."""
        model.current_lora_hash = "str([['/loras/y.safetensors', 1.0, 1.0, False]])"
        for patcher in mc_memory.model_patchers(model):
            patcher.patches = {"a": ["lora-y"]}
            patcher.patches_uuid = "uuid-rebuilt"

    def test_a_rebuilt_state_over_a_blob_is_taken_away_and_the_generation_stopped(self, blob, host):
        self.rebuilt_by_the_host(blob)
        p = make_p("a castle <lora:y:1.0>")

        reason = mc_lora_ram.before_sampling(p)

        assert reason
        for patcher in mc_memory.model_patchers(blob):
            assert patcher.patches == {}, "nothing left for the host to merge"
            assert patcher.patches_uuid == patcher.model.current_weight_patches_uuid
        assert blob.current_lora_hash == mc_lora_ram.blob_of(blob).hash
        assert host.shared.state.interrupted is True
        assert p.comments and "press Generate again" in p.comments[0]

    def test_the_next_generation_then_reloads_from_disk_whatever_it_asks(self, blob, host):
        self.rebuilt_by_the_host(blob)
        mc_lora_ram.before_sampling(make_p())

        assert mc_lora_ram.before_pass(make_p("a castle <lora:x:1.0>"))
        assert dropped(host)

    def test_a_blob_serving_its_own_set_passes_the_net(self, blob, host):
        assert mc_lora_ram.before_sampling(make_p()) == ""
        assert host.shared.state.interrupted is False

    def test_another_extensions_weight_patch_on_a_blob_passes_the_net(self, blob, host):
        """The host's record still names the baked set, so its loader took its
        early return; a patch another extension adds for one pass files the
        blob's weight as its original and puts it back. Not a LoRA on a LoRA."""
        blob.forge_objects.unet.patches = {"a": ["block-weight"]}

        assert mc_lora_ram.before_sampling(make_p()) == ""
        assert blob.forge_objects.unet.patches == {"a": ["block-weight"]}
        assert host.shared.state.interrupted is False

    def test_a_model_with_its_originals_is_not_the_nets_business(self, loaded, host):
        for patcher in mc_memory.model_patchers(loaded):
            patcher.patches_uuid = "uuid-rebuilt"

        assert mc_lora_ram.before_sampling(make_p()) == ""
        assert host.shared.state.interrupted is False


# --------------------------------------------------------------------------- #
# A blob through the RAM cache, and the two rebuild valves
# --------------------------------------------------------------------------- #


class TestABlobInTheCache:
    @pytest.fixture
    def cached(self, loaded, host, monkeypatch):
        host.shared.opts.model_chain_lora_ram = mc_lora_ram.COLD
        mc_lora_ram.after_generation(make_p("a castle <lora:x:1.0>"))
        monkeypatch.setattr(mc_memory, "loaded_size_bytes", lambda model: 8 * GB)
        monkeypatch.setattr(mc_memory, "file_size_bytes", lambda name, mods=None: 8 * GB)
        monkeypatch.setattr(mc_memory, "total_ram_bytes", lambda: 96 * GB)
        monkeypatch.setattr(mc_memory, "cache_budget_bytes", lambda: 64 * GB)
        return loaded

    def park(self, host, model, key, stage, monkeypatch):
        host.sd_models.model_data.sd_model = model
        monkeypatch.setattr(mc_memory, "_loaded_model_key", lambda: key)
        mc_memory._stash_current(stage=stage)
        return mc_memory._cache.get(key)

    def test_a_blob_coming_back_to_the_stage_that_baked_it_keeps_its_state(self, cached, host, monkeypatch):
        entry = self.park(host, cached, "key-A", mc_memory.STAGE_1, monkeypatch)

        assert mc_memory._blob_barred(entry, mc_memory.STAGE_1) == ""
        assert mc_memory._restore_prepared_state(entry, mc_memory.STAGE_1) == "preserved"

    def test_a_blob_the_other_stage_would_rebuild_is_barred_not_rebuilt(self, cached, host, monkeypatch):
        entry = self.park(host, cached, "key-A", mc_memory.STAGE_1, monkeypatch)

        assert "belongs to Stage 1" in mc_memory._blob_barred(entry, mc_memory.STAGE_2)
        assert cached.current_lora_hash != mc_lora.REBUILD

    def test_reinstating_a_barred_blob_reads_stage_1_from_disk_instead(self, cached, host, monkeypatch):
        """The warm swap back to Stage 1 finds a blob whose state it may not
        trust: the entry goes, Stage 2's model is still put away, and the
        host's own reload follows."""
        entry = self.park(host, cached, "key-A", mc_memory.STAGE_1, monkeypatch)
        cached.current_lora_hash = "hash-moved"          # changed while cached
        stage_2 = make_model(lora_hash=mc_lora.NO_NETWORKS, name="B")
        host.sd_models.model_data.sd_model = stage_2
        monkeypatch.setattr(mc_memory, "_loaded_model_key", lambda: "key-B")
        monkeypatch.setattr(mc_memory, "_loading_parameters_key", lambda: "key-A")
        mc_memory._pending_restore = "A"

        assert mc_memory.reinstate_pending() is False
        assert not mc_memory._cache.has("key-A"), "dropped: the host reads it from disk"
        assert mc_memory._cache.has("key-B"), "Stage 2's model was still put away"
        assert host.sd_models.model_data.sd_model is stage_2
        assert cached.current_lora_hash == "hash-moved", "never marked for rebuilding"
        assert mc_memory._pending_restore is None

    def test_reinstating_a_trusted_blob_is_the_warm_swap_it_always_was(self, cached, host, monkeypatch):
        self.park(host, cached, "key-A", mc_memory.STAGE_1, monkeypatch)
        stage_2 = make_model(lora_hash=mc_lora.NO_NETWORKS, name="B")
        host.sd_models.model_data.sd_model = stage_2
        monkeypatch.setattr(mc_memory, "_loaded_model_key", lambda: "key-B")
        monkeypatch.setattr(mc_memory, "_loading_parameters_key", lambda: "key-A")
        mc_memory._pending_restore = "A"

        assert mc_memory.reinstate_pending() is True
        assert host.sd_models.model_data.sd_model is cached
        assert mc_lora_ram.blob_of(cached) is not None

    def test_an_ordinary_model_in_the_same_spot_is_rebuilt_as_before(self, loaded, host, monkeypatch):
        monkeypatch.setattr(mc_memory, "loaded_size_bytes", lambda model: 8 * GB)
        monkeypatch.setattr(mc_memory, "cache_budget_bytes", lambda: 64 * GB)
        entry = self.park(host, loaded, "key-A", mc_memory.STAGE_1, monkeypatch)
        loaded.current_lora_hash = "hash-moved"

        assert mc_memory._blob_barred(entry, mc_memory.STAGE_1) == ""
        assert mc_memory._restore_prepared_state(entry, mc_memory.STAGE_1) == "rebuilt"
        assert loaded.current_lora_hash == mc_lora.REBUILD

    def test_the_failed_refine_valve_drops_a_blob_rather_than_marking_it(self, cached, host):
        host.sd_models.unloaded.clear()

        assert mc_memory.invalidate_prepared_state("a Stage 2 refine pass failed") is True
        assert dropped(host)
        assert cached.current_lora_hash != mc_lora.REBUILD

    def test_invalidate_refuses_a_blob_whoever_asks(self, cached, host):
        before = cached.current_lora_hash

        assert mc_lora.invalidate(cached, "a test") is False
        assert cached.current_lora_hash == before


# --------------------------------------------------------------------------- #
# What the menu is told
# --------------------------------------------------------------------------- #


class TestStatus:
    def test_warm_with_a_merged_lora_reports_the_originals(self, loaded):
        found = mc_lora_ram.status()

        assert found["mode"] == "warm"
        assert found["merged"] is True and found["blob"] is False
        assert found["originals_bytes"] == (512 + 512 + 64) * MB
        assert found["model"] == "A"

    def test_a_blob_reports_what_it_freed(self, loaded):
        mc_lora_ram.set_mode("cold")
        found = mc_lora_ram.status()

        assert found["blob"] is True and found["merged"] is False
        assert found["originals_bytes"] == 0
        assert found["freed_bytes"] == (512 + 512 + 64) * MB

    def test_nothing_loaded_reports_nothing(self, loaded, host):
        host.sd_models.model_data.sd_model = None
        found = mc_lora_ram.status()

        assert found["merged"] is False and found["blob"] is False and found["model"] == ""
