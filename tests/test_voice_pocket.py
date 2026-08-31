"""PocketTTS's product side: what is installed, which voices exist, and the
transaction that turns a recording into one.

Five readiness states rather than one boolean, because Pocket genuinely has
five and each has a different remedy. A machine whose speech works and whose
Clone button does not is not "half installed"; it is a machine that has not
accepted an upstream licence, and everything here exists so the panel can say
which of the five it is looking at (section 24).

The other thing asserted here is that a preview is not a save. Everything up to
the registry write leaves nothing anybody can see, so a preview that failed
halfway costs one directory to undo -- and the audition somebody hears is the
one their recording actually produced rather than a re-synthesis from a voice
that has already been written down (I-PKT-17, section 26).

Nothing here imports Torch or PocketTTS. The worker is a double, because the
questions are about registries, transactions, containment and refusals rather
than about tensors.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

import mc_voice_engines as engines
import mc_voice_paths as paths
import mc_voice_pocket as pocket


def _here() -> dict:
    """One platform entry describing the machine this suite is running on.

    Without it every status test would land in "no tested runtime for this
    platform" and prove nothing about the five states underneath -- the shipped
    manifest advertises Windows, which is the release target rather than the
    machine a test runs on.
    """
    import mc_voice_models as models

    system, machine, python_version = models.current_platform()
    return {"id": "test-runner", "system": system, "machines": [machine],
            "python": python_version, "artifacts": []}


@pytest.fixture
def unpinned(voice_root, monkeypatch):
    """This platform is supported and this build has not pinned its closure.

    The honest state of a fresh checkout, and the one the refusal has to name a
    tool for rather than saying "PocketTTS cannot be installed here".
    """
    manifest = {
        "schema": 1, "version": 1, "pinned": False,
        "runtime": {"import_name": "pocket_tts", "license": "Apache-2.0",
                    "platforms": [_here()]},
        "defaults": {"model": "english", "voice": "alba"},
        "models": {"english": {"label": "PocketTTS English", "language": "en",
                               "files": [], "required_paths": [], "voices": []}},
    }
    monkeypatch.setattr(pocket, "manifest", lambda refresh=False: manifest)
    return manifest


@pytest.fixture
def installed(voice_root, monkeypatch):
    """PocketTTS reported as fully installed, on disk, with two official voices.

    Built rather than mocked wholesale: the registry, the official state files
    and the model marker are real files under a throwaway root, so the tests
    below exercise the same reads production does.
    """
    entry_root = paths.pocket_model_root("english")
    entry_root.mkdir(parents=True, exist_ok=True)
    (entry_root / "model.safetensors").write_bytes(
        b"\x08\x00\x00\x00\x00\x00\x00\x00{}")
    (entry_root / paths.INSTALLED_FILENAME).write_text(
        json.dumps({"schema": 1, "id": "english",
                    "digests": {"model.safetensors": "a" * 64}}), encoding="utf-8")
    official = paths.pocket_official_root("english")
    official.mkdir(parents=True, exist_ok=True)
    for name in ("alba", "anna"):
        (official / f"{name}.safetensors").write_bytes(b"\x08\x00\x00\x00\x00\x00\x00\x00{}")

    manifest = {
        "schema": 1, "version": 1, "pinned": True,
        "runtime": {"import_name": "pocket_tts", "license": "Apache-2.0",
                    "platforms": [_here()]},
        "defaults": {"model": "english", "voice": "alba"},
        "models": {"english": {
            "label": "PocketTTS English", "language": "en",
            "public_repo": "kyutai/pocket-tts-without-voice-cloning",
            "cloning_repo": "kyutai/pocket-tts", "revision": "main",
            "sample_rate": 24000, "recommended_temperature": 0.3,
            "files": [{"filename": "model.safetensors", "local_name": "model.safetensors",
                       "url": "https://example.invalid/model.safetensors"}],
            "required_paths": ["model.safetensors"],
            "cloning_files": [{"filename": "cloning.safetensors",
                               "local_name": "cloning.safetensors",
                               "url": "https://example.invalid/cloning.safetensors"}],
            "cloning_required_paths": ["cloning.safetensors"],
            "config": {"weights_path": "model.safetensors"},
            "voices": [
                {"id": "alba", "display_name": "Alba", "language": "en",
                 "accent": "Scottish", "license": "CC-BY-4.0",
                 "attribution": "Kyutai", "source": "kyutai/pocket-tts",
                 "artifact": {"filename": "alba.safetensors",
                              "local_name": "alba.safetensors",
                              "url": "https://example.invalid/alba.safetensors"}},
                {"id": "anna", "display_name": "Anna", "language": "en",
                 "accent": "", "license": "CC-BY-4.0", "attribution": "Kyutai",
                 "source": "kyutai/pocket-tts",
                 "artifact": {"filename": "anna.safetensors",
                              "local_name": "anna.safetensors",
                              "url": "https://example.invalid/anna.safetensors"}}],
        }},
    }
    # A runtime marker rather than a patched status object, so ``_status``
    # computes the five readiness fields the way production does -- including
    # ``cloning_ready``, which is derived from two of the others and would have
    # been silently wrong under a wrapper that set them afterwards.
    runtime = paths.pocket_runtime_manifest()
    runtime.parent.mkdir(parents=True, exist_ok=True)
    runtime.write_text(json.dumps({"schema": 1, "closure": "", "platform": "test-runner",
                                   "pocket_version": "3.0.2", "torch_version": "2.5.0"}),
                       encoding="utf-8")
    monkeypatch.setattr(pocket, "_manifest_cache", manifest)
    monkeypatch.setattr(pocket, "manifest", lambda refresh=False: manifest)
    monkeypatch.setattr(pocket, "pinned", lambda: True)
    return manifest


@pytest.fixture
def cloning(installed, voice_root, monkeypatch):
    """The gated half installed too, so the clone transaction can run."""
    root = paths.pocket_model_root("english")
    (root / "cloning.safetensors").write_bytes(b"\x08\x00\x00\x00\x00\x00\x00\x00{}")
    (root / pocket.CLONING_MARKER).write_text(json.dumps({"schema": 1}), encoding="utf-8")
    return installed


class FakeRuntime:
    """The two calls the adapter makes into the worker, and nothing else."""

    def __init__(self):
        self.prepared = []
        self.refreshed = []
        self.forgotten = []

    def prepare_voice(self, root, voice_id, wav_bytes, seconds=0.0, audition=""):
        self.prepared.append({"root": root, "voice_id": voice_id, "seconds": seconds})
        Path(root).mkdir(parents=True, exist_ok=True)
        (Path(root) / paths.POCKET_PREVIEW_STATE_FILENAME).write_bytes(
            b"\x08\x00\x00\x00\x00\x00\x00\x00{}")
        return {"sample_rate": 24000, "audition_ms": 40, "state_bytes": 24,
                "audio": b"RIFFfake"}

    def refresh_catalog(self, voices, forget=()):
        self.refreshed.append(dict(voices))
        self.forgotten.extend(list(forget))
        return len(voices)

    def stop(self, reason=""):
        pass

    def status(self):
        return {"busy": False, "draining": False, "interrupt_mode": "drain_unit"}

    def engine(self):
        return {"loaded": False, "defaults": {}}

    def declared_interrupt_mode(self):
        return "drain_unit"


@pytest.fixture
def worker(monkeypatch):
    import sys

    found = FakeRuntime()
    monkeypatch.setitem(sys.modules, "mc_voice_pocket_runtime", found)
    return found


# --------------------------------------------------------------------------- #
# What the facade asks
# --------------------------------------------------------------------------- #


class TestTheAdapterContract:
    """Section 8. The facade asks; it no longer infers."""

    def test_it_declares_what_it_can_do(self, host, voice_root):
        found = pocket.capabilities()
        assert set(found) == {"clone_preview", "rebuild", "engine_settings",
                              "starter_voices", "voice_lab", "interrupt_mode"}
        assert found["clone_preview"] is True
        assert found["rebuild"] is True
        assert found["engine_settings"] is True
        # Neither of these, and neither is an oversight: Pocket starts with
        # official voices so the wall starter voices were built for is not
        # there, and its noise clamp and EOS threshold are not style axes
        # (section 29.5).
        assert found["starter_voices"] is False
        assert found["voice_lab"] is False

    def test_t_pkt_eng_11_it_declares_drain_unit_and_the_others_declare_cancel(
            self, host, voice_root):
        """The one capability that changes what Stop is allowed to promise."""
        import mc_voice_kokoro as kokoro
        import mc_voice_sopro as sopro

        assert pocket.capabilities()["interrupt_mode"] == "drain_unit"
        assert kokoro.capabilities()["interrupt_mode"] == "cancel"
        assert sopro.capabilities()["interrupt_mode"] == "cancel"

    def test_its_refusals_are_declared_rather_than_guessed(self, host, voice_root):
        assert pocket.refusals() == (pocket.PocketError,)
        assert engines.EngineError in engines.refusals("pocket")
        assert pocket.PocketError in engines.refusals("pocket")

    def test_the_public_status_has_the_common_subset_every_engine_answers_with(
            self, host, voice_root):
        found = pocket.public_status()
        for name in ("installed", "ready", "message", "worker_resident", "engine_busy",
                     "draining", "interrupt_mode", "block"):
            assert name in found, name

    def test_the_clone_hints_come_from_the_engine_rather_than_the_page(self, host,
                                                                       voice_root):
        """GATE P-CLONE-1. A number baked into JavaScript is a number that goes
        stale the first time somebody measures it."""
        found = pocket.clone_hints()
        assert found["min_seconds"] == pocket.MIN_REFERENCE_SECONDS
        assert found["ideal_seconds"] == pocket.IDEAL_REFERENCE_SECONDS
        assert found["max_seconds"] == pocket.MAX_REFERENCE_SECONDS
        # Below upstream's own thirty-second truncation ceiling, on purpose.
        assert found["max_seconds"] < 30


# --------------------------------------------------------------------------- #
# Status
# --------------------------------------------------------------------------- #


class TestFiveStatesRatherThanOneBoolean:
    def test_a_fresh_machine_is_not_ready_and_says_why(self, host, voice_root):
        found = pocket.status()
        assert found.ready is False
        assert found.runtime_message
        assert found.message

    def test_an_unpinned_build_says_so_and_names_the_tool(self, host, unpinned):
        """Section 23.1. An artifact this repository makes no claim about is an
        artifact it will not download, and the honest answer says which tool
        produces the claim."""
        found = pocket.status()
        assert "pin_pocket_models.py" in found.runtime_message
        assert "pin_pocket_models.py" in pocket.refusal()
        # And the folder path is still open.
        assert pocket.refusal(manual=True) == ""

    def test_speech_is_ready_without_cloning_access(self, host, installed, worker):
        """Section 23.3. "Pocket speaks" and "Pocket clones" are two states, and
        gating speech on an upstream licence nobody accepted would refuse the
        feature over the half of it that is optional."""
        found = pocket.status()
        assert found.speech_model_ready is True
        assert found.official_voices_ready is True
        assert found.ready is True
        assert found.cloning_ready is False
        assert "gated repository" in found.cloning_message
        assert "Official PocketTTS voices work without this" in found.cloning_message

    def test_with_the_gated_half_installed_cloning_is_ready(self, host, cloning, worker):
        found = pocket.status()
        assert found.ready is True
        assert found.cloning_ready is True

    def test_reading_status_starts_nothing(self, host, voice_root):
        """Section 17, and the reason ``installed`` can be read on a poll."""
        assert engines.installed("pocket") is False
        assert pocket.status().ready is False

    def test_the_fingerprint_changes_with_precision(self, host, installed, worker):
        """Section 12. Conservative, with GATE P-VOICE-1 against it: nobody has
        established that Pocket's INT8 leaves the tensors a voice state is made
        of untouched, so a precision change marks a cached state as needing a
        rebuild rather than loading one that may not mean the same thing."""
        before = pocket.status().fingerprint
        pocket.set_engine_settings(precision_id="int8")
        assert pocket.status().fingerprint != before

    def test_the_fingerprint_changes_with_the_model(self, host, installed, worker,
                                                    monkeypatch):
        before = pocket.status().fingerprint
        monkeypatch.setattr(pocket, "model_id", lambda: "other")
        installed["models"]["other"] = dict(installed["models"]["english"])
        assert pocket.status().fingerprint != before


# --------------------------------------------------------------------------- #
# Engine settings
# --------------------------------------------------------------------------- #


class TestEngineSettingsAreGlobalAndLiveInPocketsOwnFile:
    def test_the_defaults_are_the_unquantised_ones(self, host, voice_root):
        found = pocket.engine_settings()
        assert found["precision"] == "full"
        assert found["steps"] == 1

    def test_neither_precision_claims_a_speed(self, host, voice_root):
        """GATE P-5, I-PKT-26. Upstream reports faster x86 inference for INT8
        and that may well hold, but Sopro's identical-looking setting measured
        40% the other way on the first machine anybody tried."""
        for entry in pocket.engine_settings()["precisions"]:
            assert "faster" not in entry["label"].casefold()

    def test_the_step_choices_are_tested_values_with_the_number_visible(self, host,
                                                                        voice_root):
        found = pocket.engine_settings()
        assert [entry["id"] for entry in found["step_choices"]] == list(pocket.STEP_CHOICES)
        assert found["step_choices"][0]["label"] == "Fast (default)"

    def test_there_is_no_thread_control_and_a_sentence_where_one_would_be(self, host,
                                                                          voice_root):
        """Section 16.4. Setting OMP_NUM_THREADS=8 and calling it eight Pocket
        threads would be reporting something untrue."""
        found = pocket.engine_settings()
        assert "threads" not in found
        assert "thread_choices" not in found
        assert "no supported thread-count control" in found["thread_policy"]

    def test_a_setting_is_persisted_in_pockets_own_file(self, host, voice_root, worker):
        """I-PKT-19. An option is a component on the settings page as well as a
        stored value, and Apply Settings writes the page's build-time copy back
        over whatever the panel just set."""
        pocket.set_engine_settings(sampler_steps=3)
        assert pocket.steps() == 3
        stored = json.loads(paths.pocket_settings_path().read_text(encoding="utf-8"))
        assert stored[pocket.SETTING_STEPS] == 3

    def test_a_value_the_browser_invented_is_not_believed(self, host, voice_root, worker):
        """Section 33. A number a browser sent never becomes a generation policy."""
        pocket.set_engine_settings(sampler_steps=97, precision_id="float3")
        assert pocket.steps() == pocket.STEP_DEFAULT
        assert pocket.precision() == pocket.PRECISION_DEFAULT

    def test_an_effective_change_stops_the_worker(self, host, voice_root, monkeypatch):
        """I-PKT-24. A turn must never change precision in its middle, so the
        setting is written and the *next* request starts a worker with it."""
        stopped = []
        monkeypatch.setattr(pocket, "_retire", lambda reason: stopped.append(reason))
        pocket.set_engine_settings(precision_id="int8")
        assert stopped and "precision" in stopped[0]

    def test_setting_a_value_it_already_has_stops_nothing(self, host, voice_root,
                                                          monkeypatch):
        stopped = []
        monkeypatch.setattr(pocket, "_retire", lambda reason: stopped.append(reason))
        pocket.set_engine_settings(precision_id="full", sampler_steps=1)
        assert stopped == []

    def test_the_model_id_is_reserved_from_day_one(self, host, installed, worker):
        """Section 16.3. V1 may expose only English; the architecture must not
        encode "Pocket means English" in stable storage."""
        assert pocket.model_id() == "english"
        assert "model_id" in pocket.engine_settings()


# --------------------------------------------------------------------------- #
# Voices
# --------------------------------------------------------------------------- #


class TestTheVoiceLibrary:
    def test_official_voices_come_from_the_manifest(self, host, installed, worker):
        """Section 10. A manifest read rather than a network discovery event."""
        found = pocket.official()
        assert [entry["display_name"] for entry in found] == ["Alba", "Anna"]
        assert all(entry["official"] for entry in found)

    def test_an_official_voice_cannot_be_renamed_or_deleted(self, host, installed,
                                                            worker):
        assert all(not entry["editable"] for entry in pocket.official())
        assert all(not entry["deletable"] for entry in pocket.official())
        with pytest.raises(pocket.PocketError):
            pocket.rename("pocket:official:alba", "Something Else")
        with pytest.raises(pocket.PocketError):
            pocket.delete("pocket:official:alba")

    def test_the_default_is_the_reviewed_official_voice(self, host, installed, worker):
        assert pocket.default_id() == "pocket:official:alba"
        assert pocket.default_entry()["display_name"] == "Alba"

    def test_setting_a_default_writes_pockets_registry_and_nothing_else(self, host,
                                                                        installed,
                                                                        worker):
        """Section 28. Changed immediately, and with no second copy on a Forge
        settings page that Apply could put back."""
        pocket.set_default("pocket:official:anna")
        assert pocket.default_id() == "pocket:official:anna"
        stored = json.loads(paths.pocket_registry_path().read_text(encoding="utf-8"))
        assert stored["default"] == "pocket:official:anna"

    def test_resolve_carries_the_handle_privately(self, host, installed, worker):
        """Section 8. The name every engine's entry answers to, so the shared
        turn carries one opaque thing."""
        found, entry = pocket.resolve("pocket:official:alba")
        assert found == "pocket:official:alba"
        assert entry["_handle"] == "pocket:official:alba"

    def test_t_pkt_id_2_pocket_never_resolves_another_engines_voice(self, host,
                                                                    installed, worker):
        """I-PKT-2. A Sopro id is treated as absent, so the caller falls back to
        *this* engine's default rather than crossing to another bank."""
        found, _entry = pocket.resolve("sopro:clone:abcdef")
        assert found == "pocket:official:alba"
        assert pocket.lookup("sopro:clone:abcdef") is None
        assert pocket.lookup("kokoro:official:af_heart") is None

    def test_with_no_voices_at_all_it_refuses_rather_than_crossing_engines(self, host,
                                                                           voice_root):
        with pytest.raises(pocket.PocketError) as raised:
            pocket.resolve("")
        assert "no voice to speak with" in str(raised.value)

    def test_a_name_that_could_be_a_path_is_refused(self, host, installed, worker):
        """Section 45. A display name is metadata and never a filename."""
        for name in ("../escape", "a/b", "", "x" * 200):
            with pytest.raises(pocket.PocketError):
                pocket.check_name(name)

    def test_a_duplicate_name_is_refused(self, host, installed, worker):
        with pytest.raises(pocket.PocketError):
            pocket.check_name("Alba")


class TestTheCatalogueTheWorkerIsGiven:
    def test_it_maps_stable_ids_to_verified_local_paths(self, host, installed, worker):
        """I-PKT-20. The worker is handed local paths and never a repository id,
        a URL, or anything a browser supplied."""
        found = pocket.catalog()
        assert set(found) == {"pocket:official:alba", "pocket:official:anna"}
        for entry in found.values():
            assert entry["state"].endswith(".safetensors")
            assert "://" not in entry["state"]

    def test_a_voice_whose_state_is_missing_is_absent_rather_than_broken(self, host,
                                                                        installed,
                                                                        worker):
        (paths.pocket_official_root("english") / "anna.safetensors").unlink()
        assert "pocket:official:anna" not in pocket.catalog()


# --------------------------------------------------------------------------- #
# The clone transaction
# --------------------------------------------------------------------------- #


class TestAPreviewIsNotASave:
    """I-PKT-17 and section 26."""

    def test_t_pkt_clone_2_a_preview_registers_nothing(self, host, cloning, worker,
                                                       spoken_wav):
        made = pocket.prepare_preview("Test Voice", spoken_wav(8.0, 24000))
        assert made["token"]
        assert pocket.custom() == []
        assert pocket.default_id() == "pocket:official:alba"
        assert pocket.preview_state()["pending"] is True

    def test_t_pkt_clone_3_the_state_exists_only_in_the_preview_area(self, host,
                                                                     cloning, worker,
                                                                     spoken_wav):
        made = pocket.prepare_preview("Test Voice", spoken_wav(8.0, 24000))
        root = paths.pocket_preview_dir(made["token"])
        assert (root / paths.POCKET_REFERENCE_FILENAME).is_file()
        assert (root / paths.POCKET_PREVIEW_STATE_FILENAME).is_file()
        assert not paths.pocket_clones_root().exists()

    def test_t_pkt_clone_5_save_is_the_registry_commit(self, host, cloning, worker,
                                                       spoken_wav):
        made = pocket.prepare_preview("Test Voice", spoken_wav(8.0, 24000))
        kept = pocket.save_preview(made["token"])
        assert kept["voice"]["display_name"] == "Test Voice"
        assert len(pocket.custom()) == 1
        # And the state landed under its fingerprint rather than at the root.
        identifier = pocket._uuid_of(kept["voice"]["id"])
        states = list(paths.pocket_clone_states_root(identifier).glob("*.safetensors"))
        assert len(states) == 1
        assert states[0].stem == pocket.status().fingerprint

    def test_saving_does_not_change_the_default(self, host, cloning, worker, spoken_wav):
        """Section 26.3. Saving a voice is not the same decision as speaking
        with it, and a Save that silently changed what every character sounds
        like would be a Save nobody could undo."""
        made = pocket.prepare_preview("Test Voice", spoken_wav(8.0, 24000))
        pocket.save_preview(made["token"])
        assert pocket.default_id() == "pocket:official:alba"

    def test_t_pkt_clone_8_a_wrong_token_is_refused(self, host, cloning, worker,
                                                    spoken_wav):
        pocket.prepare_preview("Test Voice", spoken_wav(8.0, 24000))
        with pytest.raises(pocket.PocketError):
            pocket.save_preview("not-the-token")

    def test_t_pkt_clone_7_a_second_preview_discards_the_first(self, host, cloning,
                                                               worker, spoken_wav):
        first = pocket.prepare_preview("First", spoken_wav(8.0, 24000))
        pocket.prepare_preview("Second", spoken_wav(8.0, 24000))
        assert not paths.pocket_preview_dir(first["token"]).exists()
        with pytest.raises(pocket.PocketError):
            pocket.save_preview(first["token"])

    def test_t_pkt_clone_6_discard_deletes_the_reference_and_the_state(self, host,
                                                                       cloning, worker,
                                                                       spoken_wav):
        made = pocket.prepare_preview("Test Voice", spoken_wav(8.0, 24000))
        root = paths.pocket_preview_dir(made["token"])
        assert pocket.discard_preview(made["token"]) is True
        assert not root.exists()
        assert pocket.preview_state()["pending"] is False

    def test_a_discard_with_a_stale_token_leaves_the_current_preview_alone(self, host,
                                                                           cloning,
                                                                           worker,
                                                                           spoken_wav):
        made = pocket.prepare_preview("Test Voice", spoken_wav(8.0, 24000))
        assert pocket.discard_preview("somebody-elses-token") is False
        assert pocket.preview_state()["pending"] is True
        assert paths.pocket_preview_dir(made["token"]).exists()

    def test_t_pkt_clone_1_an_invalid_reference_is_refused_before_anything_is_written(
            self, host, cloning, worker, silent_wav, spoken_wav):
        for bad, why in ((b"", "No recording"),
                         (b"not a wav at all", "not a WAV"),
                         (spoken_wav(2.0, 24000), "seconds long"),
                         (spoken_wav(40.0, 24000), "shorter"),
                         (silent_wav(8.0, 24000), "silent")):
            with pytest.raises(pocket.PocketError) as raised:
                pocket.prepare_preview("Test Voice", bad)
            assert why in str(raised.value)
        assert not paths.pocket_preview_root().exists() or \
            not list(paths.pocket_preview_root().iterdir())

    def test_cloning_is_refused_with_a_reason_when_the_gate_is_closed(self, host,
                                                                      installed, worker,
                                                                      spoken_wav):
        with pytest.raises(pocket.PocketError) as raised:
            pocket.prepare_preview("Test Voice", spoken_wav(8.0, 24000))
        assert "gated repository" in str(raised.value)

    def test_a_failed_preparation_leaves_no_recording_behind(self, host, cloning,
                                                             worker, spoken_wav,
                                                             monkeypatch):
        """A directory with a WAV in it and no registry entry is a recording of
        somebody that nothing will ever delete."""

        def explode(**values):
            raise RuntimeError("the model fell over")

        monkeypatch.setattr(worker, "prepare_voice", explode)
        with pytest.raises(RuntimeError):
            pocket.prepare_preview("Test Voice", spoken_wav(8.0, 24000))
        assert not paths.pocket_preview_root().exists() or \
            not list(paths.pocket_preview_root().iterdir())


class TestDeletingAndRebuilding:
    def test_t_pkt_clone_9_deleting_removes_the_recording_and_every_state(self, host,
                                                                          cloning,
                                                                          worker,
                                                                          spoken_wav):
        made = pocket.prepare_preview("Test Voice", spoken_wav(8.0, 24000))
        kept = pocket.save_preview(made["token"])
        identifier = pocket._uuid_of(kept["voice"]["id"])
        root = paths.pocket_clone_root(identifier)
        assert root.exists()
        pocket.delete(kept["voice"]["id"])
        assert not root.exists()
        assert pocket.custom() == []

    def test_deleting_a_pocket_voice_cannot_reach_another_engines_files(self, host,
                                                                        cloning,
                                                                        worker,
                                                                        spoken_wav):
        """I-PKT-3, as a filesystem fact rather than a rule."""
        sopro_root = paths.sopro_root()
        sopro_root.mkdir(parents=True, exist_ok=True)
        (sopro_root / "keepme.txt").write_text("still here", encoding="utf-8")
        made = pocket.prepare_preview("Test Voice", spoken_wav(8.0, 24000))
        kept = pocket.save_preview(made["token"])
        pocket.delete(kept["voice"]["id"])
        assert (sopro_root / "keepme.txt").read_text(encoding="utf-8") == "still here"

    def test_a_stale_state_says_rebuild_rather_than_hiding_the_voice(self, host,
                                                                     cloning, worker,
                                                                     spoken_wav):
        """Section 27. A voice whose state was prepared for another model is not
        a broken voice; it is one with a remedy."""
        made = pocket.prepare_preview("Test Voice", spoken_wav(8.0, 24000))
        kept = pocket.save_preview(made["token"])
        identifier = pocket._uuid_of(kept["voice"]["id"])
        for state in paths.pocket_clone_states_root(identifier).glob("*.safetensors"):
            state.unlink()
        entry = pocket.lookup(kept["voice"]["id"])
        assert entry["compatible"] is False
        assert entry["has_source"] is True
        assert any("Rebuild it" in line for line in pocket.warnings())

    def test_t_pkt_clone_10_rebuild_writes_the_current_fingerprints_state(self, host,
                                                                          cloning,
                                                                          worker,
                                                                          spoken_wav):
        made = pocket.prepare_preview("Test Voice", spoken_wav(8.0, 24000))
        kept = pocket.save_preview(made["token"])
        identifier = pocket._uuid_of(kept["voice"]["id"])
        for state in paths.pocket_clone_states_root(identifier).glob("*.safetensors"):
            state.unlink()
        pocket.rebuild(kept["voice"]["id"])
        states = list(paths.pocket_clone_states_root(identifier).glob("*.safetensors"))
        assert [path.stem for path in states] == [pocket.status().fingerprint]
        assert pocket.lookup(kept["voice"]["id"])["compatible"] is True

    def test_an_old_states_file_is_kept_when_a_new_one_is_built(self, host, cloning,
                                                                worker, spoken_wav):
        """Section 39. Switching the model back makes the old state usable again
        without another rebuild."""
        made = pocket.prepare_preview("Test Voice", spoken_wav(8.0, 24000))
        kept = pocket.save_preview(made["token"])
        identifier = pocket._uuid_of(kept["voice"]["id"])
        first = pocket.status().fingerprint
        pocket.set_engine_settings(precision_id="int8")
        pocket.rebuild(kept["voice"]["id"])
        found = {path.stem for path in
                 paths.pocket_clone_states_root(identifier).glob("*.safetensors")}
        assert first in found
        assert pocket.status().fingerprint in found

    def test_a_voice_with_no_retained_recording_cannot_be_rebuilt_and_says_why(
            self, host, cloning, worker, spoken_wav):
        made = pocket.prepare_preview("Test Voice", spoken_wav(8.0, 24000))
        kept = pocket.save_preview(made["token"])
        identifier = pocket._uuid_of(kept["voice"]["id"])
        (paths.pocket_clone_root(identifier) / paths.POCKET_REFERENCE_FILENAME).unlink()
        with pytest.raises(pocket.PocketError) as raised:
            pocket.rebuild(kept["voice"]["id"])
        assert "no retained recording" in str(raised.value)
        # And once its state is stale as well, the warning says the harder
        # thing: a derived state with no source is not a voice with a remedy.
        for state in paths.pocket_clone_states_root(identifier).glob("*.safetensors"):
            state.unlink()
        assert any("has to be created again" in line for line in pocket.warnings())

    def test_t_pkt_clone_11_nothing_rebuilds_from_a_status_read(self, host, cloning,
                                                                worker, spoken_wav):
        """I-PKT-26. A poll that rebuilt a voice would be a poll that started a
        model and spent a minute of somebody's CPU on a decision they did not
        make."""
        made = pocket.prepare_preview("Test Voice", spoken_wav(8.0, 24000))
        kept = pocket.save_preview(made["token"])
        identifier = pocket._uuid_of(kept["voice"]["id"])
        for state in paths.pocket_clone_states_root(identifier).glob("*.safetensors"):
            state.unlink()
        before = list(worker.prepared)
        for _poll in range(3):
            pocket.status()
            pocket.entries()
            pocket.warnings()
            pocket.public_status()
        assert worker.prepared == before


class TestNoPayloadCarriesAPath:
    def test_t_pkt_id_4_no_public_entry_names_a_file_or_a_handle(self, host, cloning,
                                                                 worker, spoken_wav):
        import mc_voice_api as api

        made = pocket.prepare_preview("Test Voice", spoken_wav(8.0, 24000))
        pocket.save_preview(made["token"])
        for entry in pocket.entries():
            found = api._public(entry)
            text = json.dumps(found)
            assert "_handle" not in found
            assert str(paths.data_root()) not in text
            assert ".safetensors" not in text

    def test_the_public_status_block_carries_no_path(self, host, cloning, worker):
        text = json.dumps(pocket.public_status())
        assert str(paths.data_root()) not in text
        assert ".safetensors" not in text
