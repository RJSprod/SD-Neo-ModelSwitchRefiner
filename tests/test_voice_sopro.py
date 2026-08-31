"""Sopro's manifest, its installation, its voice library and its containment.

T-SOPRO-1 through T-SOPRO-8 and Gate S-0, S-2, S-8 and S-9 as far as they can be
asserted without a hundred and forty megabytes of PyTorch on the machine running
the tests. What is asserted here is everything the parent process decides: which
closure is pinned, what an installation has to prove before it is promoted, what
a saved voice record contains, what a delete removes, and what the worker is
told.

The one thing these tests deliberately do not do is import Torch. Every test
below passes on a machine where Sopro has never been installed, which is the
same property ``tests/test_voice_independence.py`` asserts for the Kokoro
worker and for the same reason: the day the parent cannot be imported without
the engine is the day the WebUI stops starting.
"""

from __future__ import annotations

import json
import struct
import types

import pytest

import mc_voice_engines as engines
import mc_voice_paths as paths
import mc_voice_sopro as sopro


@pytest.fixture(autouse=True)
def _no_pending_preview():
    """No test starts or ends with a voice waiting to be saved.

    The pending preview is process-global on purpose — one at a time, so that
    there is never a second unregistered directory to reason about — which means
    a test that leaves one behind changes the answer for the next one. Cleared
    in memory rather than through ``discard_preview`` so that this never removes
    a directory outside a test's own temporary root.
    """
    sopro._preview.clear()
    yield
    sopro._preview.clear()


class TestTheManifest:
    def test_gate_s_0_every_runtime_wheel_is_pinned(self):
        """An artifact without a hash is one this repository makes no claim
        about, and the whole preparation fingerprint rests on knowing which
        bytes ran. There is no "download it and trust it" mode for the closure
        and there is no flag to add one."""
        found = sopro.manifest()
        for entry in found["runtime"]["platforms"]:
            for artifact in entry["artifacts"]:
                assert artifact["sha256"], f"{artifact['filename']} has no digest"
                assert artifact["bytes"] > 0, f"{artifact['filename']} has no byte count"

    def test_the_closure_is_windows_cpu_only(self):
        """I-9 and section 17. PyPI's Linux Torch wheels pull the whole CUDA
        closure through their dependency markers, so a Linux platform in this
        manifest would be an engine that can claim a graphics device. Adding one
        is a deliberate act with its own gate, not something that happens
        because somebody ran the pinning tool on Linux."""
        found = sopro.manifest()
        systems = {entry["system"] for entry in found["runtime"]["platforms"]}
        assert systems == {"windows"}
        for entry in found["runtime"]["platforms"]:
            names = " ".join(item["filename"] for item in entry["artifacts"])
            assert "nvidia" not in names and "cuda" not in names and "triton" not in names

    def test_the_advertised_pythons_are_an_allowlist(self):
        found = sopro.manifest()
        assert sorted(entry["python"] for entry in found["runtime"]["platforms"]) \
            == ["3.10", "3.11", "3.12", "3.13"]

    def test_the_closure_carries_sopro_torch_and_torchaudio(self):
        found = sopro.manifest()
        for entry in found["runtime"]["platforms"]:
            names = [item["filename"].split("-")[0].lower() for item in entry["artifacts"]]
            for wanted in ("sopro", "torch", "torchaudio", "numpy", "safetensors",
                           "sentencepiece", "soundfile"):
                assert wanted in names, f"{wanted} is missing from {entry['id']}"

    def test_the_closure_has_no_hugging_face_client(self):
        """Section 54, enforced by absence rather than by a flag. Production
        passes Sopro a verified local directory; a worker with no hub client
        cannot resolve a model from the Internet even if something asked it to."""
        found = sopro.manifest()
        for entry in found["runtime"]["platforms"]:
            names = " ".join(item["filename"].lower() for item in entry["artifacts"])
            assert "huggingface" not in names

    def test_the_model_bundle_names_the_files_sopro_actually_reads(self):
        """``sopro.hub.ARTIFACT_FILES``, held to by a test rather than by a
        comment, because the machine writing the manifest is not the machine
        that has the runtime."""
        entry = sopro.bundle()
        assert set(entry.required_paths) == {
            "config.json", "model.safetensors", "semantic_encoder.safetensors",
            "speaker_encoder.safetensors", "vocoder.safetensors",
            "vocoder_streaming.safetensors", "tokenizer.model"}

    def test_the_languages_offered_are_sopros_own_tags(self):
        """Section 33. Four tags plus the absence of one, and the codes are the
        ones ``sopro.text.LANGUAGE_TAGS`` declares."""
        codes = {code for code, _label in sopro.LANGUAGES}
        assert codes == {"", "en", "pt", "fr", "de"}

    def test_a_manifest_this_build_cannot_read_is_a_broken_extension(self, monkeypatch,
                                                                     tmp_path):
        broken = tmp_path / "manifest.json"
        broken.write_text(json.dumps({"schema": 99}), encoding="utf-8")
        monkeypatch.setattr(paths, "sopro_manifest_path", lambda: broken)
        sopro._manifest_cache_clear()
        with pytest.raises(sopro.SoproError):
            sopro.manifest()
        sopro._manifest_cache_clear()


class TestStatus:
    def test_reading_status_starts_nothing(self, voice_root, monkeypatch):
        """Section 17: a status route a browser polls must never be able to
        begin a download or a model load."""
        monkeypatch.setattr(sopro, "install", lambda *a, **k: pytest.fail("installed"))
        found = sopro.status()
        assert found.ready is False
        assert not (voice_root / "sopro").exists()

    def test_an_unsupported_platform_says_so_and_says_kokoro_is_unaffected(
            self, voice_root, monkeypatch):
        monkeypatch.setattr(sopro, "platform", lambda: None)
        found = sopro.status()
        assert found.platform_supported is False
        assert "Kokoro is unaffected" in found.runtime_message

    def test_t_sopro_8_the_fingerprint_covers_the_closure_and_the_model(
            self, voice_root, monkeypatch):
        """Section 15: compatibility is identified by actual closure and content,
        not by a package version alone."""
        chosen = sopro.platform() or _fake_platform()
        first = sopro._fingerprint(chosen, {"sopro_version": "2.0.5",
                                            "torch_version": "2.11.0"},
                                   {"digests": {"model.safetensors": "aaaa"}})
        same = sopro._fingerprint(chosen, {"sopro_version": "2.0.5",
                                           "torch_version": "2.11.0"},
                                  {"digests": {"model.safetensors": "aaaa"}})
        other_model = sopro._fingerprint(chosen, {"sopro_version": "2.0.5",
                                                  "torch_version": "2.11.0"},
                                         {"digests": {"model.safetensors": "bbbb"}})
        other_torch = sopro._fingerprint(chosen, {"sopro_version": "2.0.5",
                                                  "torch_version": "2.12.0"},
                                         {"digests": {"model.safetensors": "aaaa"}})
        assert first == same
        assert first != other_model
        assert first != other_torch

    def test_precision_is_not_in_the_fingerprint(self, voice_root):
        """A voice prepared at full precision is correct at INT8 and the
        reverse: quantization touches the autoregressive blocks, and the
        encoders that produce a voice's conditioning are untouched by it. What
        precision *does* invalidate is the warmed prompt state, which is keyed
        separately inside the worker."""
        chosen = sopro.platform() or _fake_platform()
        installed = {"sopro_version": "2.0.5", "torch_version": "2.11.0"}
        found = sopro._fingerprint(chosen, installed, {})
        assert "full" not in found and "int8" not in found


class TestTheWorkerEnvironment:
    def test_i_9_the_worker_can_see_no_graphics_device(self):
        found = sopro.worker_environment()
        assert found["CUDA_VISIBLE_DEVICES"] == ""
        assert found["HIP_VISIBLE_DEVICES"] == ""
        assert found["ROCR_VISIBLE_DEVICES"] == ""

    def test_the_thread_caps_match_the_released_policy(self):
        """Section 20: a Kokoro measured at four threads beside a Sopro that
        silently took every logical core is not a comparison, it is two
        different experiments."""
        from sopro_worker import worker as protocol

        found = sopro.worker_environment()
        assert found["OMP_NUM_THREADS"] == str(protocol.INTRAOP_THREADS)
        assert found["MKL_NUM_THREADS"] == str(protocol.INTRAOP_THREADS)

    def test_the_worker_is_told_it_is_offline(self):
        found = sopro.worker_environment()
        assert found["HF_HUB_OFFLINE"] == "1"

    def test_no_user_site_reaches_the_isolated_runtime(self):
        assert sopro.worker_environment()["PYTHONNOUSERSITE"] == "1"


class TestEngineSettings:
    def test_only_tested_values_are_accepted(self, host, voice_root):
        sopro.set_engine_settings(precision_id="int8", solver_steps=99, chunk=7)
        found = sopro.engine_settings()
        assert found["precision"] == "int8"
        assert found["steps"] in sopro.STEP_CHOICES
        assert found["chunk_frames"] in sopro.CHUNK_CHOICES

    def test_changing_one_stops_the_worker(self, host, voice_root, monkeypatch):
        """Section 34: precision is chosen at model load, and the steps and the
        chunk size decide which warmed prompt states are valid at all. Restarting
        is one cold load; a worker whose caches no longer match its settings is a
        class of bug nobody can hear the start of."""
        import mc_voice_sopro_runtime as runtime

        stopped = []
        monkeypatch.setattr(runtime, "stop", lambda reason="": stopped.append(reason))
        sopro.set_engine_settings(precision_id="int8")
        assert stopped and "precision" in stopped[0]

    def test_changing_one_does_not_touch_kokoro(self, host, voice_root, kokoro_bundle,
                                                voice_registry, monkeypatch):
        """I-3, and the reason the option names are separate."""
        import mc_voice_profile as kokoro_profile
        import mc_voice_sopro_runtime as runtime

        monkeypatch.setattr(runtime, "stop", lambda reason="": None)
        before = kokoro_profile.stored()
        sopro.set_engine_settings(precision_id="int8", solver_steps=4, chunk=128)
        assert kokoro_profile.stored() == before


class TestTheSolverStepDefault:
    """The main quality control this engine has, and which end of it ships.

    Upstream's reviewed V2 default is 2 and it documents larger values as a
    quality/compute trade. This build shipped at 2 as well, which is the setting
    that makes a *streaming* engine keep up on an unknown machine — and the
    thing people actually report is that a cloned voice does not sound enough
    like them. It now ships at the top of the range.
    """

    def test_an_untouched_installation_gets_the_top_of_the_range(self, host, voice_root):
        assert sopro.steps() == max(sopro.STEP_CHOICES)
        assert sopro.STEP_DEFAULT == max(sopro.STEP_CHOICES)

    def test_the_default_is_named_rather_than_an_index_into_the_choices(self):
        """It was ``STEP_CHOICES[0]``, so the shipped default was a property of
        the *order* of a tuple: reordering it for the UI would have silently
        changed what every untouched installation runs.

        Read out of the function's own source rather than out of the file, so a
        mention in a docstring elsewhere explaining the history does not make
        this pass or fail.
        """
        import inspect

        body = inspect.getsource(sopro.steps)
        assert "STEP_CHOICES[0]" not in body, body
        assert "STEP_DEFAULT" in body, body

    def test_a_stored_choice_still_wins(self, host, voice_root, monkeypatch):
        import mc_voice_sopro_runtime as runtime

        monkeypatch.setattr(runtime, "stop", lambda reason="": None)
        sopro.set_engine_settings(solver_steps=2)
        assert sopro.steps() == 2

    def test_a_stored_value_that_is_not_offered_falls_back_to_the_default(
            self, host, voice_root):
        """Not to the first choice, which is what made this worth naming: a
        corrupt or older settings file must land on the shipped setting rather
        than on whichever value happens to be written first."""
        sopro._remember(sopro.OPT_STEPS, 5)
        assert sopro.steps() == sopro.STEP_DEFAULT


class TestTheCpuThreadSetting:
    """The lever the validation sweep exists to inform, made selectable.

    It began as an environment variable, which was the wrong place: Precision is
    already a user control that changes compute, RAM and which warmed caches
    survive, and thread count is the same kind of thing. What I-12 forbids is
    the *code* picking a number from a measured real-time factor — a person
    reading a table and choosing a row is what "one measured, fixed policy"
    asks for. Asking that person to set an environment variable on Windows is
    asking them not to bother.
    """

    def test_the_default_is_the_released_policy(self, host, voice_root):
        from sopro_worker import worker as protocol

        assert sopro.intraop_threads() == protocol.INTRAOP_THREADS

    def test_a_chosen_count_is_kept(self, host, voice_root, monkeypatch):
        import mc_voice_sopro_runtime as runtime

        monkeypatch.setattr(runtime, "stop", lambda reason="": None)
        monkeypatch.setattr(sopro, "thread_choices", lambda: (2, 4, 8))
        sopro.set_engine_settings(threads=8)
        assert sopro.intraop_threads() == 8
        assert sopro.engine_settings()["threads"] == 8

    def test_a_count_that_is_not_offered_is_refused(self, host, voice_root, monkeypatch):
        """Checked against the offerable list rather than against a range, so a
        number a browser invented cannot become a thread count."""
        import mc_voice_sopro_runtime as runtime

        monkeypatch.setattr(runtime, "stop", lambda reason="": None)
        monkeypatch.setattr(sopro, "thread_choices", lambda: (2, 4, 8))
        sopro.set_engine_settings(threads=999)
        assert sopro.intraop_threads() != 999

    def test_changing_it_stops_the_worker(self, host, voice_root, monkeypatch):
        """The policy is applied before the model loads, so a running worker
        cannot adopt it. Restarting is one cold load."""
        import mc_voice_sopro_runtime as runtime

        stopped = []
        monkeypatch.setattr(runtime, "stop", lambda reason="": stopped.append(reason))
        monkeypatch.setattr(sopro, "thread_choices", lambda: (2, 4, 8))
        sopro.set_engine_settings(threads=8)
        assert stopped and "thread" in stopped[0], stopped

    def test_the_choices_never_exceed_the_machine(self, monkeypatch):
        """Offering more threads than cores would put a row in the dropdown
        that is reliably worse than the one above it."""
        monkeypatch.setattr(sopro.os, "cpu_count", lambda: 4)
        assert max(sopro.thread_choices()) <= 4

    def test_the_released_policy_is_always_reachable(self, monkeypatch):
        """A dropdown that could not express the shipped configuration would be
        one somebody could not get back to."""
        from sopro_worker import worker as protocol

        monkeypatch.setattr(sopro.os, "cpu_count", lambda: 1)
        assert protocol.INTRAOP_THREADS in sopro.thread_choices()

    def test_the_chosen_count_reaches_the_worker(self, host, voice_root, monkeypatch):
        """The whole point. A setting the child never hears about would be a
        dropdown that changes a number in a file.

        Both channels carry it: the worker reads the count back out of the
        environment and hands it to Torch, and OpenMP sizes its pool from the
        same value — so the two cannot disagree.
        """
        import mc_voice_sopro_runtime as runtime
        from sopro_worker import worker as protocol

        monkeypatch.setattr(runtime, "stop", lambda reason="": None)
        monkeypatch.setattr(sopro, "thread_choices", lambda: (2, 4, 8))
        sopro.set_engine_settings(threads=8)
        found = sopro.worker_environment()
        assert found[protocol.OVERRIDE_INTRAOP] == "8"
        assert found["OMP_NUM_THREADS"] == "8"

    def test_choosing_the_released_number_does_not_read_as_an_override(
            self, host, voice_root, monkeypatch):
        """Selecting 4 deliberately is the shipped policy, and every log line
        should keep saying so."""
        import mc_voice_sopro_runtime as runtime
        from sopro_worker import worker as protocol

        monkeypatch.setattr(runtime, "stop", lambda reason="": None)
        monkeypatch.setattr(sopro, "thread_choices", lambda: (2, 4, 8))
        sopro.set_engine_settings(threads=protocol.INTRAOP_THREADS)
        monkeypatch.setenv(protocol.OVERRIDE_INTRAOP,
                           sopro.worker_environment()[protocol.OVERRIDE_INTRAOP])
        _intraop, _interop, overridden = protocol.effective_policy(note=False)
        assert overridden is False

    def test_a_sweep_override_still_wins_over_the_setting(self, host, voice_root,
                                                          monkeypatch):
        """The sweep measures configurations this installation has not chosen,
        so its number has to beat the stored one for the row to mean anything."""
        import mc_voice_sopro_runtime as runtime
        from sopro_worker import worker as protocol

        monkeypatch.setattr(runtime, "stop", lambda reason="": None)
        monkeypatch.setattr(sopro, "thread_choices", lambda: (2, 4, 8))
        sopro.set_engine_settings(threads=8)
        monkeypatch.setenv(protocol.OVERRIDE_INTRAOP, "2")
        assert sopro.intraop_threads() == 2


def _tone_energy(samples, rate, freq, width=4096):
    """Energy at one frequency, by Hann-windowed correlation.

    Written out rather than reached for from NumPy so that this measurement
    runs wherever the tests do, and so that it exercises no part of what it is
    measuring.
    """
    import cmath
    import math as _math

    start = max(0, len(samples) // 2 - width // 2)
    chunk = samples[start:start + width]
    assert len(chunk) == width, (
        f"the analysis window is {len(chunk)} of {width}; the test signal is too short "
        f"and this measurement would silently read zero")
    total = 0j
    for index in range(width):
        shaped = 0.5 - 0.5 * _math.cos(2 * _math.pi * index / width)
        total += chunk[index] * shaped * cmath.exp(-2j * _math.pi * freq * index / rate)
    return abs(total) ** 2


def _tone(freq, rate, seconds=2.0, amplitude=12000):
    import array
    import math as _math

    return array.array("h", [int(amplitude * _math.sin(2 * _math.pi * freq * t / rate))
                             for t in range(int(rate * seconds))])


def _decibels(got, reference):
    import math as _math

    return 10 * _math.log10(got / reference + 1e-30)


def _linear_resample(samples, rate, target):
    """The resampler this build shipped with, kept so the test can fail.

    Four lines, no filter. It is here rather than in the module because a
    regression test for "the resampler aliases" needs something that aliases to
    prove it can tell the difference -- otherwise a threshold of -40 dB is a
    number nobody has ever seen fail.
    """
    import array

    wanted = int(len(samples) * target / float(rate))
    ratio = len(samples) / float(max(1, wanted))
    out = array.array("h", bytes(2 * wanted))
    for index in range(wanted):
        position = index * ratio
        left = int(position)
        right = min(left + 1, len(samples) - 1)
        weight = position - left
        out[index] = int(samples[left] * (1.0 - weight) + samples[right] * weight)
    return out


class TestTheResampler:
    """Section 25, and the largest single thing that was wrong with a clone.

    Almost every recording anybody clones from is 44.1 or 48 kHz and Sopro wants
    24. The resampler that shipped was linear interpolation with no anti-alias
    filter at all, which does not attenuate what it cannot represent -- it folds
    it. Everything a microphone captured between 12 and 24 kHz came back mirrored
    into 0-12 kHz on top of the voice, at full amplitude: mic self-noise, room
    hiss, and sibilance, which is precisely where "s" and "sh" and "t" live.

    The conditioning is computed from these samples, so a clone inherited it
    permanently. Measured below rather than described.
    """

    #: A tone above the new Nyquist folds to (24000 - freq). 15 kHz lands at
    #: 9 kHz, in the middle of the speech band, which is what makes this audible
    #: rather than academic.
    SOURCE_RATE = 48000
    ABOVE_NYQUIST = 15000
    FOLDS_TO = 9000

    def test_the_shipped_linear_resampler_folded_it_back_at_full_amplitude(self):
        """The defect, pinned. Not a claim about the past: this runs the old
        four lines and measures them, so the threshold in the next test is one
        that has actually been seen to fail."""
        tone = _tone(self.ABOVE_NYQUIST, self.SOURCE_RATE)
        reference = _tone_energy(tone, self.SOURCE_RATE, self.ABOVE_NYQUIST)
        folded = _tone_energy(_linear_resample(tone, self.SOURCE_RATE, 24000),
                              24000, self.FOLDS_TO)
        assert _decibels(folded, reference) > -3.0, (
            "the old resampler was supposed to alias at roughly full amplitude")

    def test_a_tone_above_the_new_nyquist_is_gone(self):
        tone = _tone(self.ABOVE_NYQUIST, self.SOURCE_RATE)
        reference = _tone_energy(tone, self.SOURCE_RATE, self.ABOVE_NYQUIST)
        folded = _tone_energy(sopro._resample(tone, self.SOURCE_RATE, 24000),
                              24000, self.FOLDS_TO)
        assert _decibels(folded, reference) < -60.0, _decibels(folded, reference)

    def test_the_pure_python_path_filters_too(self):
        """Narrower, because the same twenty-four zero crossings in a loop would
        take most of a minute -- but a different universe from no filter."""
        tone = _tone(self.ABOVE_NYQUIST, self.SOURCE_RATE, seconds=0.5)
        reference = _tone_energy(tone, self.SOURCE_RATE, self.ABOVE_NYQUIST)
        count = int(len(tone) * 24000 / self.SOURCE_RATE)
        folded = _tone_energy(
            sopro._resample_slowly(tone, self.SOURCE_RATE, 24000, count),
            24000, self.FOLDS_TO)
        assert _decibels(folded, reference) < -40.0, _decibels(folded, reference)

    def test_speech_band_content_comes_through_untouched(self):
        """A filter that removed the aliasing by removing the top of the voice
        would measure just as well on the test above and be worse than the bug."""
        for freq in (200, 1000, 3000, 6000):
            tone = _tone(freq, self.SOURCE_RATE)
            reference = _tone_energy(tone, self.SOURCE_RATE, freq)
            kept = _tone_energy(sopro._resample(tone, self.SOURCE_RATE, 24000),
                                24000, freq)
            assert abs(_decibels(kept, reference)) < 0.5, (freq,
                                                           _decibels(kept, reference))

    def test_44100_is_handled_as_well_as_48000(self):
        """The commoner rate, and the harder one: the ratio is not an integer,
        so every output sample lands between two input ones."""
        tone = _tone(15000, 44100)
        reference = _tone_energy(tone, 44100, 15000)
        # 44100 -> 24000 folds 15 kHz to 24000 - 15000 = 9 kHz as well.
        folded = _tone_energy(sopro._resample(tone, 44100, 24000), 24000, 9000)
        assert _decibels(folded, reference) < -60.0, _decibels(folded, reference)

    def test_the_length_is_the_ratio(self):
        import array

        source = array.array("h", [0] * 48000)
        assert len(sopro._resample(source, 48000, 24000)) == 24000

    def test_upsampling_does_not_alias_either(self):
        """16 kHz recordings exist, and going up must not mirror the top of the
        source band into the new headroom."""
        tone = _tone(7000, 16000)
        reference = _tone_energy(tone, 16000, 7000)
        kept = _tone_energy(sopro._resample(tone, 16000, 24000), 24000, 7000)
        assert abs(_decibels(kept, reference)) < 1.0, _decibels(kept, reference)

    def test_a_rate_that_needs_no_change_is_left_alone(self):
        import array

        source = array.array("h", [1, -1, 300, -300] * 100)
        assert sopro._resample(source, 24000, 24000) is source


class TestReferenceValidation:
    def test_a_recording_shorter_than_the_window_is_refused(self, voice_root, spoken_wav):
        with pytest.raises(sopro.SoproError) as raised:
            sopro.normalize_reference(spoken_wav(2.0, 24000))
        assert "5" in str(raised.value)

    def test_the_refusals_still_call_this_engine_what_they_always_called_it(
            self, voice_root, spoken_wav, silent_wav):
        """The decoder moved into :mod:`mc_voice_reference` when a third engine
        that clones from a recording arrived, and it takes the engine's name
        from an envelope. Taking it from ``LABEL`` would have turned every one
        of these sentences from "Sopro" into "Sopro V2" -- moving code is not a
        reason to change what a user reads."""
        for bad in (spoken_wav(2.0, 24000), spoken_wav(40.0, 24000),
                    silent_wav(10.0, 24000), b"not a wav at all, not even close"):
            with pytest.raises(sopro.SoproError) as raised:
                sopro.normalize_reference(bad)
            assert "Sopro V2" not in str(raised.value), str(raised.value)
        assert sopro.ENVELOPE.engine == "Sopro"

    def test_a_recording_longer_than_the_window_is_refused(self, voice_root, spoken_wav):
        with pytest.raises(sopro.SoproError) as raised:
            sopro.normalize_reference(spoken_wav(40.0, 24000))
        assert "20" in str(raised.value)

    def test_t_sopro_6_a_silent_recording_is_refused(self, voice_root, silent_wav):
        """Section 25's conservative level check. It catches a muted input and a
        wrong device, which are the two ways somebody records twenty seconds of
        nothing and cannot tell why the clone came out wrong."""
        with pytest.raises(sopro.SoproError) as raised:
            sopro.normalize_reference(silent_wav(10.0, 24000))
        assert "silent" in str(raised.value)

    def test_something_that_is_not_a_wav_is_refused(self, voice_root):
        with pytest.raises(sopro.SoproError):
            sopro.normalize_reference(b"not a wav at all, not even close")

    def test_an_empty_body_is_refused(self, voice_root):
        with pytest.raises(sopro.SoproError):
            sopro.normalize_reference(b"")

    def test_an_oversized_body_is_refused_before_it_is_parsed(self, voice_root):
        with pytest.raises(sopro.SoproError):
            sopro.normalize_reference(b"RIFF" + b"\x00" * (sopro.MAX_REFERENCE_BYTES + 1))

    def test_a_good_recording_becomes_canonical_mono_24k(self, voice_root, spoken_wav):
        found, seconds = sopro.normalize_reference(spoken_wav(8.0, 16000))
        assert found[:4] == b"RIFF"
        (channels,) = struct.unpack_from("<H", found, 22)
        (rate,) = struct.unpack_from("<I", found, 24)
        assert channels == 1
        assert rate == sopro.TARGET_RATE
        assert 7.5 < seconds < 8.5

    def test_the_validator_is_sopros_own_rather_than_storytimes(self):
        """I-6. ``mc_voice_clone`` is Kokoro's Storytime path with Storytime's
        three-to-a-hundred-and-twenty-second window; an engine whose validation
        lived in the other engine's module would be an engine that could not
        change its own rules."""
        import mc_voice_clone

        assert sopro.MIN_REFERENCE_SECONDS != mc_voice_clone.MIN_REFERENCE_SECONDS
        assert sopro.MAX_REFERENCE_SECONDS != mc_voice_clone.MAX_REFERENCE_SECONDS



def reference_wav(seconds: float, rate: int, channels: int, encoding: int, bits: int,
                  extensible: bool = False, level: float = 0.35) -> bytes:
    """One tone, written the way the named encoding writes it.

    Built here rather than taken from the ``spoken_wav`` fixture because the
    whole point of these tests is the container: the fixture only knows how to
    write the one format that already worked.
    """
    import math
    import struct as _struct

    frames = int(rate * seconds)
    samples = [level * math.sin(2 * math.pi * 220 * index / rate)
               for index in range(frames) for _ in range(channels)]
    if encoding == 1 and bits == 8:
        body = bytes(max(0, min(255, int(value * 127) + 128)) for value in samples)
    elif encoding == 1 and bits == 16:
        body = b"".join(_struct.pack("<h", int(value * 32767)) for value in samples)
    elif encoding == 1 and bits == 24:
        body = b"".join(int(value * 8388607).to_bytes(3, "little", signed=True)
                        for value in samples)
    elif encoding == 1 and bits == 32:
        body = b"".join(_struct.pack("<i", int(value * 2147483647)) for value in samples)
    elif encoding == 3 and bits == 32:
        body = b"".join(_struct.pack("<f", value) for value in samples)
    elif encoding == 3 and bits == 64:
        body = b"".join(_struct.pack("<d", value) for value in samples)
    else:
        # Something this build has no codec for. The bytes do not matter; the
        # header is what is being refused.
        body = bytes(len(samples) * max(1, bits // 8))

    block = channels * max(1, bits // 8)
    tag = 0xFFFE if extensible else encoding
    fmt = _struct.pack("<HHIIHH", tag, channels, rate, rate * block, block, bits)
    if extensible:
        # cbSize, valid bits, channel mask, then the SubFormat GUID whose first
        # two bytes carry the real format tag.
        fmt += _struct.pack("<HHI", 22, bits, 3 if channels == 2 else 4)
        fmt += (_struct.pack("<H", encoding) + b"\x00\x00"
                + b"\x00\x00\x10\x00\x80\x00\x00\xaa\x00\x38\x9b\x71")
    chunks = (b"fmt " + _struct.pack("<I", len(fmt)) + fmt
              + b"data" + _struct.pack("<I", len(body)) + body)
    return b"RIFF" + _struct.pack("<I", 4 + len(chunks)) + b"WAVE" + chunks


class TestTheEncodingsAPersonActuallyHas:
    """The reference decoder against the files recorders and editors produce.

    The regression: this accepted ``WAVE_FORMAT_PCM`` at exactly sixteen bits
    and nothing else, so a recording that had been through any editor -- 24-bit,
    32-bit float, or ordinary 16-bit PCM wrapped in ``WAVE_FORMAT_EXTENSIBLE``,
    which is what a writer reaches for the moment a file is not plain -- was
    refused with a sentence about what Sopro wanted and no word about what the
    file was. Sopro still wants mono 24 kHz PCM16; this function's job is to
    produce that, and narrowing a sample is the same work as the downmix and
    the resample it already did.
    """

    CASES = [
        ("16-bit PCM", dict(encoding=1, bits=16)),
        ("16-bit PCM in EXTENSIBLE", dict(encoding=1, bits=16, extensible=True)),
        ("24-bit PCM", dict(encoding=1, bits=24)),
        ("32-bit PCM", dict(encoding=1, bits=32)),
        ("32-bit float", dict(encoding=3, bits=32)),
        ("32-bit float in EXTENSIBLE", dict(encoding=3, bits=32, extensible=True)),
        ("64-bit float", dict(encoding=3, bits=64)),
        ("8-bit PCM", dict(encoding=1, bits=8)),
    ]

    @pytest.mark.parametrize("label,shape", CASES, ids=[name for name, _ in CASES])
    def test_it_becomes_canonical_mono_24k(self, voice_root, label, shape):
        found, seconds = sopro.normalize_reference(
            reference_wav(8.0, 48000, 1, **shape))

        assert found[:4] == b"RIFF"
        (encoding,) = struct.unpack_from("<H", found, 20)
        (channels,) = struct.unpack_from("<H", found, 22)
        (rate,) = struct.unpack_from("<I", found, 24)
        (bits,) = struct.unpack_from("<H", found, 34)
        assert (encoding, channels, rate, bits) == (1, 1, sopro.TARGET_RATE, 16)
        assert 7.5 < seconds < 8.5, label

    @pytest.mark.parametrize("label,shape", CASES, ids=[name for name, _ in CASES])
    def test_the_audio_survives_the_conversion(self, voice_root, label, shape):
        """A decoder that returned silence would pass the shape checks above and
        then fail Sopro's level gate -- so this asserts the level directly, at
        roughly the 0.35 the tone was written at."""
        found, _seconds = sopro.normalize_reference(reference_wav(8.0, 48000, 1, **shape))

        import array

        samples = array.array("h")
        samples.frombytes(found[44:])
        peak = max(abs(value) for value in samples) / 32768.0
        assert 0.25 < peak < 0.45, f"{label} decoded to a peak of {peak:.3f}"

    def test_a_stereo_float_recording_is_downmixed(self, voice_root):
        found, seconds = sopro.normalize_reference(
            reference_wav(8.0, 44100, 2, encoding=3, bits=32))

        (channels,) = struct.unpack_from("<H", found, 22)
        assert channels == 1
        assert 7.5 < seconds < 8.5

    def test_a_compressed_recording_is_refused_and_named(self, voice_root):
        """Refusing is right -- decoding it needs a codec this module should not
        grow -- but the sentence has to say what the file is, which is the half
        that lets somebody fix it."""
        with pytest.raises(sopro.SoproError) as raised:
            sopro.normalize_reference(reference_wav(8.0, 44100, 1, encoding=0x0011, bits=4))

        assert "IMA ADPCM" in str(raised.value)

    def test_an_unknown_encoding_is_named_by_its_number(self, voice_root):
        with pytest.raises(sopro.SoproError) as raised:
            sopro.normalize_reference(reference_wav(8.0, 44100, 1, encoding=0x2001, bits=16))

        assert "0x2001" in str(raised.value)

    def test_a_float_recording_carrying_nan_does_not_crash(self, voice_root):
        """A NaN is what a badly-written float WAV puts where a sample goes, and
        `int(nan)` raises. Silence is the only honest substitute."""
        import struct as _struct

        good = reference_wav(8.0, 24000, 1, encoding=3, bits=32)
        head = good[:44]
        body = bytearray(good[44:])
        body[0:4] = _struct.pack("<f", float("nan"))
        found, _seconds = sopro.normalize_reference(bytes(head) + bytes(body))

        assert found[:4] == b"RIFF"

    def test_a_float_recording_above_unity_is_clipped_rather_than_wrapped(self,
                                                                          voice_root):
        """A float WAV may exceed 1.0. Scaling that by 32767 and truncating to
        int16 wraps a loud peak round to a loud peak of the opposite sign, which
        is an audible click in the reference the whole voice is built from."""
        found, _seconds = sopro.normalize_reference(
            reference_wav(8.0, 24000, 1, encoding=3, bits=32, level=1.8))

        import array

        samples = array.array("h")
        samples.frombytes(found[44:])
        assert max(samples) == 32767
        assert min(samples) >= -32768
        # Wrapping shows up as neighbouring samples at opposite extremes.
        assert not any(abs(samples[i] - samples[i + 1]) > 40000
                       for i in range(len(samples) - 1))

    def test_extensible_with_a_truncated_header_is_refused(self, voice_root):
        """The SubFormat GUID is where the real tag lives; a fmt chunk too short
        to hold one is malformed, not a licence to guess PCM."""
        good = reference_wav(8.0, 24000, 1, encoding=1, bits=16)
        broken = bytearray(good)
        struct.pack_into("<H", broken, 20, 0xFFFE)
        with pytest.raises(sopro.SoproError) as raised:
            sopro.normalize_reference(bytes(broken))

        assert "malformed" in str(raised.value)


class TestNames:
    def test_a_name_that_is_not_a_name_is_refused(self, voice_root):
        for bad in ("", "   ", "../etc/passwd", "a" * 100, "voice\x00name"):
            with pytest.raises(sopro.SoproError):
                sopro.check_name(bad)

    def test_an_ordinary_name_is_accepted(self, voice_root):
        assert sopro.check_name("  Rebecca (test) ") == "Rebecca (test)"


class TestTheVoiceLibrary:
    def test_an_empty_library_has_no_default_and_does_not_invent_one(self, host, voice_root):
        """Section 53: Sopro selected with no Sopro voice means Voice says a
        voice must be created. It never resolves to Kokoro."""
        assert sopro.entries() == []
        assert sopro.default_id() == ""
        with pytest.raises(sopro.SoproError):
            sopro.resolve()

    def test_sopro_has_no_built_in_speakers(self):
        """Section 8: bundled example voices, if ever shipped, must be
        explicitly licensed and represented separately from user clones. Until
        then this is empty rather than a placeholder."""
        assert sopro.official() == []

    def test_the_default_survives_an_apply_on_the_settings_page(self, host, voice_root,
                                                                 fake_sopro_worker,
                                                                 spoken_wav):
        """The reported bug, reproduced.

        Forge's "Apply settings" writes every component on the settings page
        back into the option store, and the browser's copy of an option is
        stamped when the page is built -- so it knows nothing about a "Set as
        Default" pressed since. The default lived in an option, so setting one
        and then changing anything else on that page put the old value quietly
        back, and Conversation went on speaking with the first voice in the list.

        It lives in the registry file now, which no settings form can reach.
        """
        first = sopro.create("First", spoken_wav(9.0, 24000), "en")["voice"]
        second = sopro.create("Second", spoken_wav(9.0, 24000), "en")["voice"]
        sopro.set_default(second["id"])

        # What an Apply does: the stale component value goes back into the store.
        host.shared.opts.set(sopro.OPT_VOICE, first["id"])

        assert sopro.default_id() == second["id"]
        assert sopro.resolve("")[0] == second["id"]

    def test_a_default_set_by_an_older_build_is_still_honoured(self, host, voice_root,
                                                              fake_sopro_worker,
                                                              spoken_wav):
        """Migration: the option is read once, when the file has nothing."""
        sopro.create("First", spoken_wav(9.0, 24000), "en")
        second = sopro.create("Second", spoken_wav(9.0, 24000), "en")["voice"]

        stored = sopro._read()
        stored["default"] = ""
        sopro._write(stored)
        host.shared.opts.set(sopro.OPT_VOICE, second["id"])

        assert sopro.default_id() == second["id"]

    def test_starter_voices_are_not_bundled_speakers(self):
        """The distinction section 8 actually draws.

        A starter voice is *made* on the machine it runs on, from Kokoro reading
        a passage, and what it leaves behind is an ordinary Sopro clone with its
        own retained reference. Nothing ships with the extension, so the "no
        built-in speakers" rule above is untouched by it."""
        assert sopro.official() == []
        assert [name for _sid, name in sopro.STARTER_VOICES]

    def test_the_starter_list_shrinks_as_they_are_made(self, host, voice_root,
                                                       fake_sopro_worker, spoken_wav,
                                                       monkeypatch):
        import mc_voice_models as models
        import mc_voice_registry as registry
        import mc_voice_runtime as kokoro

        monkeypatch.setattr(models, "status",
                            lambda: types.SimpleNamespace(tts_ready=True))
        monkeypatch.setattr(registry, "resolve", lambda voice_id="": (3, {"id": voice_id}))
        monkeypatch.setattr(kokoro, "synthesize",
                            lambda text, sid=0, profile=None: spoken_wav(16.0, 24000))

        assert len(sopro.starter_names()) == len(sopro.STARTER_VOICES)
        made = sopro.create_starter_voice()

        assert made["created"] == sopro.STARTER_VOICES[0][1]
        assert made["remaining"] == len(sopro.STARTER_VOICES) - 1
        assert len(sopro.starter_names()) == len(sopro.STARTER_VOICES) - 1
        # And what it left behind is an ordinary voice, not a special kind.
        found = [entry for entry in sopro.entries()
                 if entry["display_name"] == made["created"]]
        assert found and found[0]["id"].startswith("sopro:clone:")
        assert found[0]["editable"] and found[0]["deletable"]

    def test_a_starter_voice_is_read_by_kokoro_and_prepared_by_sopro(self, host,
                                                                     voice_root,
                                                                     fake_sopro_worker,
                                                                     spoken_wav,
                                                                     monkeypatch):
        """The one place the two engines touch, and it is a creation step: the
        reference is Kokoro's, the voice that comes out is Sopro's own and keeps
        working if Kokoro is removed afterwards."""
        import mc_voice_models as models
        import mc_voice_registry as registry
        import mc_voice_runtime as kokoro

        asked = []
        monkeypatch.setattr(models, "status",
                            lambda: types.SimpleNamespace(tts_ready=True))
        monkeypatch.setattr(registry, "resolve",
                            lambda voice_id="": (asked.append(voice_id) or (3, {})))
        monkeypatch.setattr(kokoro, "synthesize",
                            lambda text, sid=0, profile=None: (asked.append(len(text))
                                                               or spoken_wav(16.0, 24000)))
        sopro.create_starter_voice()

        assert asked[0] == "official:" + sopro.STARTER_VOICES[0][0]
        assert asked[1] > 200, "the reference passage is too short to condition on"

    def test_starter_voices_say_so_when_kokoro_is_missing(self, host, voice_root,
                                                          fake_sopro_worker, monkeypatch):
        """Rather than failing somewhere inside a runtime that was never asked
        to be installed."""
        import mc_voice_models as models

        monkeypatch.setattr(models, "status",
                            lambda: types.SimpleNamespace(tts_ready=False))
        with pytest.raises(sopro.SoproError, match="Kokoro"):
            sopro.create_starter_voice()

    def test_asking_again_when_they_all_exist_is_not_an_error(self, host, voice_root,
                                                              fake_sopro_worker,
                                                              spoken_wav, monkeypatch):
        import mc_voice_models as models
        import mc_voice_registry as registry
        import mc_voice_runtime as kokoro

        monkeypatch.setattr(models, "status",
                            lambda: types.SimpleNamespace(tts_ready=True))
        monkeypatch.setattr(registry, "resolve", lambda voice_id="": (3, {}))
        monkeypatch.setattr(kokoro, "synthesize",
                            lambda text, sid=0, profile=None: spoken_wav(16.0, 24000))
        for _ in sopro.STARTER_VOICES:
            sopro.create_starter_voice()

        found = sopro.create_starter_voice()
        assert found == {"created": "", "remaining": 0}

    def test_t_sopro_7_a_voice_is_created_renamed_defaulted_and_deleted(
            self, host, voice_root, fake_sopro_worker, spoken_wav):
        made = sopro.create("Rebecca", spoken_wav(9.0, 24000), "en")
        voice = made["voice"]
        assert voice["id"].startswith("sopro:clone:")
        assert voice["display_name"] == "Rebecca"
        assert voice["language"] == "en"
        assert made["audio"][:4] == b"RIFF"

        # The first voice becomes the default, because an installation with one
        # voice and no default is one where Auto Speak silently does nothing.
        assert sopro.default_id() == voice["id"]

        renamed = sopro.rename(voice["id"], "Bex")
        assert renamed["display_name"] == "Bex"
        assert renamed["id"] == voice["id"], "a rename changed the stable id"

        root = paths.sopro_voice_root(voice["id"].split(":")[-1])
        assert (root / paths.SOPRO_REFERENCE_FILENAME).is_file()
        assert (root / paths.SOPRO_PRODUCTION_FILENAME).is_file()
        assert (root / paths.SOPRO_LAB_FILENAME).is_file()

        sopro.delete(voice["id"])
        assert sopro.entries() == []
        assert not root.exists(), "a deleted voice left files behind"
        assert sopro.default_id() == ""

    def test_t_sopro_5_prompt_state_is_never_a_file(self, host, voice_root,
                                                    fake_sopro_worker, spoken_wav):
        """I-11. Warmed streaming state is worker cache with a bound on it, and
        the durable files are conditioning plus a scalar."""
        made = sopro.create("Rebecca", spoken_wav(9.0, 24000), "")
        root = paths.sopro_voice_root(made["voice"]["id"].split(":")[-1])
        names = {item.name for item in root.iterdir()}
        assert names == {paths.SOPRO_REFERENCE_FILENAME, paths.SOPRO_PRODUCTION_FILENAME,
                         paths.SOPRO_PRODUCTION_META, paths.SOPRO_LAB_FILENAME}
        assert not any("prompt" in name or "kv" in name or "session" in name
                       for name in names)

    def test_a_failed_preparation_registers_nothing_and_keeps_no_recording(
            self, host, voice_root, fake_sopro_worker, spoken_wav, monkeypatch):
        """Section 27. A directory with a recording in it and no registry entry
        is a recording of somebody that nothing will ever delete."""
        import mc_voice_sopro_runtime as runtime

        monkeypatch.setattr(runtime, "prepare_voice",
                            lambda **kwargs: (_ for _ in ()).throw(
                                runtime.SoproRuntimeError("no")))
        with pytest.raises(runtime.SoproRuntimeError):
            sopro.create("Rebecca", spoken_wav(9.0, 24000), "")
        assert sopro.entries() == []
        assert not list(paths.sopro_voices_root().glob("*/reference.wav"))

    def test_preparing_a_preview_writes_no_registry_entry(
            self, host, voice_root, fake_sopro_worker, spoken_wav):
        """The whole point. Create used to build the voice, write it into the
        registry, make it the default if it was the first, and *then* play the
        audition — so hearing it was a receipt rather than a decision."""
        made = sopro.prepare_preview("Rebecca", spoken_wav(9.0, 24000), "")
        assert made["token"]
        assert made["audio"], "the preview came back with nothing to listen to"
        assert sopro.entries() == [], "the voice was saved before anybody heard it"
        assert sopro.preview_state()["pending"] is True

    def test_the_prepared_voice_is_on_disk_but_not_in_the_registry(
            self, host, voice_root, fake_sopro_worker, spoken_wav):
        """Section 23's seam. Everything up to the registry write leaves nothing
        anybody can see, which is why the failure path has always been one
        rmtree — a prepared voice with no entry is not half-saved."""
        sopro.prepare_preview("Rebecca", spoken_wav(9.0, 24000), "")
        assert list(paths.sopro_voices_root().glob("*/reference.wav"))
        assert sopro.entries() == []

    def test_saving_the_preview_is_what_registers_it(
            self, host, voice_root, fake_sopro_worker, spoken_wav):
        made = sopro.prepare_preview("Rebecca", spoken_wav(9.0, 24000), "")
        kept = sopro.save_preview(made["token"])
        assert kept["voice"]["display_name"] == "Rebecca"
        assert [entry["display_name"] for entry in sopro.entries()] == ["Rebecca"]
        assert sopro.preview_state()["pending"] is False

    def test_discarding_takes_the_recording_with_it(
            self, host, voice_root, fake_sopro_worker, spoken_wav):
        """A directory holding a recording of somebody, with nothing in the
        registry pointing at it, is one nothing would ever remove."""
        sopro.prepare_preview("Rebecca", spoken_wav(9.0, 24000), "")
        assert sopro.discard_preview() is True
        assert sopro.entries() == []
        assert not list(paths.sopro_voices_root().glob("*/reference.wav"))
        assert sopro.preview_state()["pending"] is False

    def test_a_second_preview_discards_the_first(
            self, host, voice_root, fake_sopro_worker, spoken_wav):
        """One at a time. Two unregistered directories is two things to reason
        about and nobody asked for the second one."""
        first = sopro.prepare_preview("Rebecca", spoken_wav(9.0, 24000), "")
        sopro.prepare_preview("Ada", spoken_wav(9.0, 24000), "")
        assert len(list(paths.sopro_voices_root().glob("*/reference.wav"))) == 1
        with pytest.raises(sopro.SoproError):
            sopro.save_preview(first["token"])

    def test_saving_with_the_wrong_token_is_refused(
            self, host, voice_root, fake_sopro_worker, spoken_wav):
        """A browser sitting on an old panel must not be able to save a voice
        the user has already replaced with another preview."""
        sopro.prepare_preview("Rebecca", spoken_wav(9.0, 24000), "")
        with pytest.raises(sopro.SoproError):
            sopro.save_preview("not-the-token")
        assert sopro.entries() == []
        assert sopro.preview_state()["pending"] is True, \
            "a refused save threw the preview away"

    def test_saving_when_nothing_is_pending_says_so(self, host, voice_root):
        with pytest.raises(sopro.SoproError) as raised:
            sopro.save_preview("anything")
        assert "no voice waiting" in str(raised.value)

    def test_discarding_someone_else_s_preview_is_refused(
            self, host, voice_root, fake_sopro_worker, spoken_wav):
        sopro.prepare_preview("Rebecca", spoken_wav(9.0, 24000), "")
        assert sopro.discard_preview("not-the-token") is False
        assert sopro.preview_state()["pending"] is True

    def test_the_first_saved_voice_becomes_the_default_and_a_preview_does_not(
            self, host, voice_root, fake_sopro_worker, spoken_wav):
        """An installation with one voice and no default is one where Auto
        Speak silently does nothing — but a voice nobody has kept must not be
        what the whole installation speaks with."""
        sopro.prepare_preview("Rebecca", spoken_wav(9.0, 24000), "")
        assert not str(sopro._read().get("default") or "")
        made = sopro.prepare_preview("Ada", spoken_wav(9.0, 24000), "")
        kept = sopro.save_preview(made["token"])
        assert sopro._read().get("default") == kept["voice"]["id"]

    def test_shutdown_throws_a_pending_preview_away(
            self, host, voice_root, fake_sopro_worker, spoken_wav, monkeypatch):
        """The decision was "not yet", and a WebUI that closed on "not yet" has
        answered it. Otherwise the directory outlives the process that knew
        about it and nothing ever removes it."""
        import mc_voice_sopro_runtime as runtime

        sopro.prepare_preview("Rebecca", spoken_wav(9.0, 24000), "")
        assert list(paths.sopro_voices_root().glob("*/reference.wav"))
        runtime.shutdown()
        assert not list(paths.sopro_voices_root().glob("*/reference.wav"))
        assert sopro.preview_state()["pending"] is False

    def test_create_still_prepares_and_keeps_in_one_call(
            self, host, voice_root, fake_sopro_worker, spoken_wav):
        """Starter voices have nobody to ask: they are Kokoro reading a fixed
        script, so there is no recording anybody chose and nothing to audition
        before deciding."""
        made = sopro.create("Rebecca", spoken_wav(9.0, 24000), "")
        assert made["voice"]["display_name"] == "Rebecca"
        assert made["audio"]
        assert sopro.preview_state()["pending"] is False

    def test_a_voice_directory_is_a_uuid_and_never_a_display_name(
            self, host, voice_root, fake_sopro_worker, spoken_wav):
        """Section 57: every clone path is built from a server-generated
        identifier, and a display name never becomes a filename.

        The name here is a *legal* one that happens to contain the punctuation a
        filename would notice. A name that is not legal is refused a step
        earlier by :func:`check_name`, which is the test below this one; this
        one is about what happens to the names that get through.
        """
        made = sopro.create("Rebecca (test) & co.", spoken_wav(9.0, 24000), "")
        found = [item.name for item in paths.sopro_voices_root().iterdir()
                 if item.is_dir()]
        assert len(found) == 1
        assert found[0].isalnum() and len(found[0]) == 32
        assert "Rebecca" not in found[0]
        assert made["voice"]["display_name"] == "Rebecca (test) & co."

    def test_a_name_that_could_be_a_path_never_reaches_the_filesystem(
            self, host, voice_root, fake_sopro_worker, spoken_wav):
        with pytest.raises(sopro.SoproError):
            sopro.create("../../etc/passwd", spoken_wav(9.0, 24000), "")
        assert not list(paths.sopro_voices_root().glob("*")) \
            if paths.sopro_voices_root().exists() else True

    def test_the_catalogue_hides_a_voice_prepared_by_another_build(
            self, host, voice_root, fake_sopro_worker, spoken_wav):
        """Section 15: a stale asset is never guessed compatible. It is absent
        from the worker's catalogue rather than read against the wrong schema."""
        made = sopro.create("Rebecca", spoken_wav(9.0, 24000), "")
        assert made["voice"]["id"] in sopro.catalog()

        stored = sopro._read()
        stored["voices"][0]["fingerprint"] = "somethingelse00"
        sopro._write(stored)
        assert sopro.catalog() == {}
        assert any("different Sopro build" in warning for warning in sopro.warnings())

    def test_a_stale_voice_with_a_recording_offers_a_rebuild(
            self, host, voice_root, fake_sopro_worker, spoken_wav):
        made = sopro.create("Rebecca", spoken_wav(9.0, 24000), "")
        stored = sopro._read()
        stored["voices"][0]["fingerprint"] = "somethingelse00"
        sopro._write(stored)

        found = sopro.lookup(made["voice"]["id"])
        assert found["compatible"] is False
        assert found["has_source"] is True

        sopro.rebuild(made["voice"]["id"])
        assert sopro.lookup(made["voice"]["id"])["compatible"] is True

    def test_deleting_a_sopro_voice_cannot_touch_a_kokoro_bank(
            self, host, voice_root, fake_sopro_worker, spoken_wav, kokoro_bundle,
            voice_registry):
        """Section 8: deletion is explicit and engine-local."""
        made = sopro.create("Rebecca", spoken_wav(9.0, 24000), "")
        before = {entry["id"] for entry in voice_registry.entries()}
        sopro.delete(made["voice"]["id"])
        assert {entry["id"] for entry in voice_registry.entries()} == before

    def test_a_voice_from_the_other_engine_is_not_this_librarys(self, host, voice_root):
        assert sopro.lookup("kokoro:official:af_heart") is None
        with pytest.raises(sopro.SoproError):
            sopro.set_default("kokoro:official:af_heart")


class TestUninstalling:
    def test_it_refuses_while_sopro_is_the_selected_engine(self, host, voice_root):
        engines.select("sopro")
        with pytest.raises(sopro.SoproError):
            sopro.uninstall()

    def test_it_keeps_the_voices(self, host, voice_root, fake_sopro_worker, spoken_wav):
        """Somebody who removes a hundred and forty megabytes of Torch to free
        space has not asked to throw away the recordings they made of
        themselves, and those recordings are what a reinstall rebuilds from."""
        made = sopro.create("Rebecca", spoken_wav(9.0, 24000), "")
        paths.sopro_runtime_root().mkdir(parents=True, exist_ok=True)
        engines.select("kokoro")
        sopro.uninstall()
        assert not paths.sopro_runtime_root().exists()
        assert sopro.lookup(made["voice"]["id"]) is not None


def _fake_platform():
    """A stand-in closure for a test running where the manifest has no entry."""
    import mc_voice_models as models

    return models.RuntimePlatform(identifier="windows-x86_64-cp311", system="windows",
                                  machines=("amd64",), python="3.11", artifacts=())


# --------------------------------------------------------------------------- #
# Installing it
# --------------------------------------------------------------------------- #


def _wheel(directory, name: str, version: str = "1.0"):
    """The smallest thing this installer will unpack, so a real build is cheap."""
    import zipfile

    path = directory / f"{name}-{version}-py3-none-any.whl"
    dist = f"{name}-{version}.dist-info"
    with zipfile.ZipFile(path, "w") as wheel:
        wheel.writestr(f"{name}/__init__.py", "VALUE = 1\n")
        wheel.writestr(f"{dist}/METADATA",
                       f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n")
        wheel.writestr(f"{dist}/WHEEL",
                       "Wheel-Version: 1.0\nGenerator: test\nRoot-Is-Purelib: true\n"
                       "Tag: py3-none-any\n")
        wheel.writestr(f"{dist}/RECORD",
                       f"{name}/__init__.py,,\n{dist}/METADATA,,\n{dist}/WHEEL,,\n"
                       f"{dist}/RECORD,,\n")
    return path


def _closure(tmp_path, names=("sopro", "torch")):
    """A staged wheel directory and the platform that names it."""
    import hashlib

    import mc_voice_models as models

    wheels = tmp_path / "wheels"
    wheels.mkdir(parents=True, exist_ok=True)
    artifacts = []
    for name in names:
        built = _wheel(wheels, name)
        payload = built.read_bytes()
        artifacts.append(models.Artifact(
            filename=built.name, local_name=built.name,
            url=f"https://example.invalid/{built.name}",
            size=len(payload), sha256=hashlib.sha256(payload).hexdigest()))
    chosen = models.RuntimePlatform(identifier="test", system="windows",
                                    machines=("amd64",), python="3.11",
                                    artifacts=tuple(artifacts))
    return wheels, chosen


class TestBuildingTheIsolatedRuntime:
    def test_the_wheels_are_unpacked_without_a_package_manager(self, tmp_path, monkeypatch):
        """R2-2, and the correction to a real failure: pip can report success,
        exit zero, and have installed into somebody's ``PIP_TARGET`` or user
        site, leaving the runtime empty. An installer that resolves nothing
        cannot resolve something from an index."""
        import subprocess

        import mc_voice_models as models

        def explode(*args, **kwargs):
            raise AssertionError("the Sopro installer ran a subprocess")

        monkeypatch.setattr(subprocess, "run", explode)
        wheels, chosen = _closure(tmp_path)
        staging = tmp_path / "staging"
        staging.mkdir()

        sopro._build_environment(staging, wheels, chosen)
        target = models.site_packages(staging / "env")
        assert (target / "sopro").is_dir()
        assert (target / "torch").is_dir()

    def test_it_is_built_without_pip(self, tmp_path, monkeypatch):
        """``ensurepip`` is one of the likelier things to be broken on the
        embedded and relocated Pythons a WebUI is launched from."""
        import venv

        seen = {}
        original = venv.EnvBuilder.__init__

        def record(self, *args, **kwargs):
            seen.update(kwargs)
            return original(self, *args, **kwargs)

        monkeypatch.setattr(venv.EnvBuilder, "__init__", record)
        wheels, chosen = _closure(tmp_path)
        staging = tmp_path / "staging"
        staging.mkdir()
        sopro._build_environment(staging, wheels, chosen)
        assert seen.get("with_pip") is False

    def test_an_unpack_that_leaves_sopro_or_torch_missing_is_refused(self, tmp_path):
        """"The installer said it worked" is precisely the claim that turned out
        to be worthless, so both are checked by name."""
        wheels, chosen = _closure(tmp_path, names=("sopro",))
        staging = tmp_path / "staging"
        staging.mkdir()
        with pytest.raises(sopro.SoproError, match="torch is not in"):
            sopro._build_environment(staging, wheels, chosen)

    def test_a_missing_wheel_is_named(self, tmp_path):
        import mc_voice_models as models

        wheels = tmp_path / "wheels"
        wheels.mkdir()
        chosen = models.RuntimePlatform(
            identifier="test", system="windows", machines=("amd64",), python="3.11",
            artifacts=(models.Artifact(filename="gone.whl", local_name="gone.whl",
                                       url="https://example.invalid/gone.whl",
                                       size=1, sha256="0" * 64),))
        staging = tmp_path / "staging"
        staging.mkdir()
        with pytest.raises(sopro.SoproError, match="gone.whl"):
            sopro._build_environment(staging, wheels, chosen)


class TestInstallingFromAFolder:
    def test_a_file_with_the_right_name_and_the_wrong_contents_is_refused(self, tmp_path,
                                                                          voice_root):
        """The manual path is not the trusting path. The runtime wheels are
        pinned in this repository, so what you supply is checked against a hash
        committed here exactly as a download is."""
        import hashlib

        import mc_voice_models as models

        folder = tmp_path / "downloads"
        folder.mkdir()
        (folder / "torch.whl").write_bytes(b"not a wheel at all")
        artifact = models.Artifact(filename="torch.whl", local_name="torch.whl",
                                   url="https://example.invalid/torch.whl",
                                   size=len(b"the real one"),
                                   sha256=hashlib.sha256(b"the real one").hexdigest())
        with pytest.raises(sopro.SoproError, match="not what this extension"):
            sopro._adopt([artifact], folder, tmp_path / "staged", lambda _t: None,
                         "the Sopro runtime")

    def test_a_file_that_is_simply_absent_is_named(self, tmp_path, voice_root):
        import mc_voice_models as models

        folder = tmp_path / "downloads"
        folder.mkdir()
        artifact = models.Artifact(filename="torch.whl", local_name="torch.whl",
                                   url="https://example.invalid/torch.whl",
                                   size=4, sha256="0" * 64)
        with pytest.raises(sopro.SoproError, match="torch.whl is not in"):
            sopro._adopt([artifact], folder, tmp_path / "staged", lambda _t: None,
                         "the Sopro runtime")

    def test_a_folder_that_is_not_one_is_refused(self, tmp_path, voice_root):
        with pytest.raises(sopro.SoproError, match="not a folder"):
            sopro._adopt([], tmp_path / "nowhere", tmp_path / "staged", lambda _t: None,
                         "the Sopro runtime")

    def test_a_good_file_is_copied_and_its_digest_recorded(self, tmp_path, voice_root):
        """Recorded because these become the constant the *next* install of that
        bundle is checked against."""
        import hashlib

        import mc_voice_models as models

        folder = tmp_path / "downloads"
        folder.mkdir()
        payload = b"the real one"
        (folder / "model.safetensors").write_bytes(payload)
        artifact = models.Artifact(filename="model.safetensors",
                                   local_name="model.safetensors",
                                   url="https://example.invalid/model.safetensors",
                                   size=None, sha256=None)
        staged = tmp_path / "staged"
        found = sopro._adopt([artifact], folder, staged, lambda _t: None, "the Sopro model")
        assert (staged / "model.safetensors").read_bytes() == payload
        assert found["model.safetensors"] == hashlib.sha256(payload).hexdigest()


class TestTheDownloadedModelIsCheckedForShape:
    def test_an_html_error_page_under_a_safetensors_name_is_refused(self, tmp_path,
                                                                    voice_root):
        """A proxy that answers every request with an error page produces files
        of plausible size under the right names, and "Sopro will not load" three
        minutes later is a much worse report than "that file is not a
        safetensors"."""
        staging = tmp_path / "staged"
        staging.mkdir()
        for name in sopro.bundle().required_paths:
            (staging / name).write_bytes(b"<html>403 Forbidden</html>" * 40)
        with pytest.raises(sopro.SoproError, match="safetensors|readable JSON"):
            sopro._sanity_check(staging, sopro.bundle())

    def test_an_empty_file_is_refused(self, tmp_path, voice_root):
        staging = tmp_path / "staged"
        staging.mkdir()
        for name in sopro.bundle().required_paths:
            (staging / name).write_bytes(b"")
        with pytest.raises(sopro.SoproError, match="empty"):
            sopro._sanity_check(staging, sopro.bundle())
