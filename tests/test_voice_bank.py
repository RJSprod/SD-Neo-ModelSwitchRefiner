"""The Kokoro voice bank: metadata, layout, transactions, and rollback.

This is the module that makes a cloned voice reachable by a runtime that has no
concept of registering one. Everything it does rests on three facts read out of
sherpa-onnx's own loader rather than assumed:

    the bank is one flat block of float32, ``n_speakers`` blocks of
        ``style_rows x style_dim``;
    the loader refuses to start unless the float count matches the model's
        metadata exactly;
    a speaker id past ``n_speakers`` is *not* an error -- sherpa logs a warning
        and uses speaker 0.

The third is why half of these tests exist. A bank whose metadata did not take
produces a clone that works perfectly and sounds like somebody else, which is
the failure mode that would survive every test that only checked for errors.
"""

from __future__ import annotations

import struct

import pytest

import mc_voice_bank as bank
import mc_voice_paths as paths

SPEAKER_BYTES = 510 * 256 * 4


def block_of(raw: bytes, sid: int) -> bytes:
    return raw[sid * SPEAKER_BYTES:(sid + 1) * SPEAKER_BYTES]


def marker_of(raw: bytes, sid: int) -> float:
    """Which voice ended up at this speaker id.

    The fixture fills every official speaker's block with its own id as a float,
    so "is af_heart still at 3" is a question with a numeric answer.
    """
    return struct.unpack_from("<f", block_of(raw, sid))[0]


class TestTheModelMetadata:
    def test_metadata_is_read_out_of_the_file_and_not_a_manifest(self, kokoro_bundle):
        found = bank.read_metadata(kokoro_bundle / "model.onnx")
        assert found["n_speakers"] == "53"
        assert found["style_dim"] == "510,1,256"
        assert bank.speaker_capacity(kokoro_bundle / "model.onnx") == 53
        assert bank.style_shape(kokoro_bundle / "model.onnx") == (510, 256)

    def test_t_bank_2_the_derived_model_declares_the_extended_count(self, kokoro_bundle,
                                                                    voice_root):
        """The gate the whole custom-voice design stands on. Without it the bank
        is refused for being the wrong size -- and with a *partial* version of
        it, a custom sid would silently speak as speaker 0."""
        target = voice_root / "derived.onnx"
        bank.derive_model(kokoro_bundle / "model.onnx", target, 85)
        assert bank.speaker_capacity(target) == 85

    def test_the_derivation_changes_one_string_and_nothing_else(self, kokoro_bundle,
                                                               voice_root):
        target = voice_root / "derived.onnx"
        bank.derive_model(kokoro_bundle / "model.onnx", target, 85)
        before = bank.read_metadata(kokoro_bundle / "model.onnx")
        after = bank.read_metadata(target)
        assert set(before) == set(after)
        assert {key: value for key, value in after.items() if key != "n_speakers"} == \
               {key: value for key, value in before.items() if key != "n_speakers"}

    def test_the_derivation_is_reproducible(self, kokoro_bundle, voice_root):
        """Section 51: generated reproducibly from the pinned upstream model.
        Same input, same bytes, same hash -- which is what makes a rebuild after
        a crash something that can be compared rather than hoped about."""
        first = bank.derive_model(kokoro_bundle / "model.onnx", voice_root / "a.onnx", 85)
        second = bank.derive_model(kokoro_bundle / "model.onnx", voice_root / "b.onnx", 85)
        assert first == second
        assert (voice_root / "a.onnx").read_bytes() == (voice_root / "b.onnx").read_bytes()

    def test_a_speaker_count_whose_length_changes_still_works(self, kokoro_bundle,
                                                              voice_root):
        """"53" and "85" are both two characters, which would let a
        byte-for-byte substitution pass this test's easy case and fail the day
        somebody chose a different capacity."""
        target = voice_root / "big.onnx"
        bank.derive_model(kokoro_bundle / "model.onnx", target, 1234)
        assert bank.speaker_capacity(target) == 1234

    def test_a_model_with_no_speaker_count_is_refused(self, voice_root):
        """Adding the key would be inventing a claim about a file rather than
        adjusting one."""
        import conftest

        plain = voice_root / "plain.onnx"
        plain.parent.mkdir(parents=True, exist_ok=True)
        plain.write_bytes(conftest._onnx_model({"model_type": "kokoro"}))
        with pytest.raises(bank.BankError, match="speaker count"):
            bank.derive_model(plain, voice_root / "no.onnx", 85)


class TestVoicepackValidation:
    def test_t_bank_4_a_510_row_voicepack_is_accepted(self, voice_root, voicepack):
        source = voicepack(voice_root / "a.bin", rows=510)
        assert len(bank.read_voicepack(source)) == SPEAKER_BYTES

    def test_t_bank_5_a_511_row_voicepack_is_normalized_to_510(self, voice_root, voicepack):
        """Section 54. Storytime writes 510 or 511 rows; sherpa's Kokoro v1.0
        addresses rows below 510 and no more, because ``style_dim[0]`` is also
        its maximum token length. The extra row is real and is simply not
        reachable by this runtime."""
        source = voicepack(voice_root / "b.bin", rows=511)
        found = bank.read_voicepack(source)
        assert len(found) == SPEAKER_BYTES
        assert found == source.read_bytes()[:SPEAKER_BYTES]

    def test_t_bank_6_any_other_row_count_is_refused(self, voice_root, voicepack):
        source = voicepack(voice_root / "c.bin", rows=300)
        with pytest.raises(bank.BankError, match="rows"):
            bank.read_voicepack(source)

    def test_a_length_that_is_not_whole_rows_is_refused(self, voice_root):
        source = voice_root / "d.bin"
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_bytes(b"\x00" * 1234)
        with pytest.raises(bank.BankError, match="rows"):
            bank.read_voicepack(source)

    def test_an_empty_or_missing_voicepack_is_refused(self, voice_root):
        (voice_root / "e.bin").parent.mkdir(parents=True, exist_ok=True)
        (voice_root / "e.bin").write_bytes(b"")
        with pytest.raises(bank.BankError, match="empty"):
            bank.read_voicepack(voice_root / "e.bin")
        with pytest.raises(bank.BankError, match="could not be read"):
            bank.read_voicepack(voice_root / "nothing.bin")

    def test_t_bank_7_nan_and_infinity_are_refused(self, voice_root):
        """A style vector with a NaN in it produces silence, or noise, or a
        native crash, depending on the day. The moment to find out is before it
        is written into the bank everything loads from."""
        for value in (float("nan"), float("inf")):
            source = voice_root / "f.bin"
            source.parent.mkdir(parents=True, exist_ok=True)
            source.write_bytes(struct.pack("<f", 0.5) * (510 * 256 - 1)
                               + struct.pack("<f", value))
            with pytest.raises(bank.BankError, match="not numbers"):
                bank.read_voicepack(source)


class TestBuilding:
    def test_t_bank_1_official_voices_keep_their_upstream_order(self, kokoro_bundle):
        bank.build({})
        bank.promote()
        raw = paths.bank_voices().read_bytes()
        for sid in (0, 3, 27, 52):
            assert marker_of(raw, sid) == pytest.approx(float(sid))

    def test_t_bank_3_the_bank_is_exactly_the_size_the_metadata_declares(self,
                                                                        kokoro_bundle):
        found = bank.build({})
        assert found["speakers"] == 53 + bank.CUSTOM_CAPACITY
        assert found["bytes"] == found["speakers"] * SPEAKER_BYTES
        bank.promote()
        assert bank.speaker_capacity(paths.bank_model()) == found["speakers"]
        assert paths.bank_voices().stat().st_size == found["bytes"]

    def test_a_custom_voice_lands_at_its_own_speaker_id(self, kokoro_bundle, voice_root,
                                                        voicepack):
        source = voicepack(voice_root / "clone.bin", rows=510, value=99.0)
        bank.build({2: str(source)})
        bank.promote()
        raw = paths.bank_voices().read_bytes()
        assert bank.custom_base() == 53
        assert bank.sid_for_slot(2) == 55
        assert marker_of(raw, 55) == pytest.approx(99.0)

    def test_t_bank_12_an_empty_slot_holds_a_real_voice_and_is_never_shown(self,
                                                                          kokoro_bundle):
        """Zeros would be a structurally valid bank that produces silence or
        noise if anything ever addressed the slot, and "valid but wrong" is the
        failure hardest to notice."""
        bank.build({})
        bank.promote()
        raw = paths.bank_voices().read_bytes()
        # af_heart is speaker 3 in the pinned map, and is the filler.
        assert marker_of(raw, 53) == pytest.approx(3.0)
        assert bank.capacity()["slots"] == {}

    def test_t_bank_13_every_reserved_slot_can_be_filled(self, kokoro_bundle, voice_root,
                                                         voicepack):
        slots = {}
        for slot in range(bank.CUSTOM_CAPACITY):
            slots[slot] = str(voicepack(voice_root / f"c{slot}.bin", value=float(slot) + 100))
        bank.build(slots)
        bank.promote()
        raw = paths.bank_voices().read_bytes()
        for slot in range(bank.CUSTOM_CAPACITY):
            assert marker_of(raw, bank.sid_for_slot(slot)) == pytest.approx(float(slot) + 100)

    def test_the_build_is_deterministic(self, kokoro_bundle, voice_root, voicepack):
        source = voicepack(voice_root / "clone.bin")
        assert bank.build({0: str(source)})["hash"] == bank.build({0: str(source)})["hash"]

    def test_a_bundle_whose_model_and_bank_disagree_is_refused(self, kokoro_bundle):
        (kokoro_bundle / "voices.bin").write_bytes(
            (kokoro_bundle / "voices.bin").read_bytes()[:SPEAKER_BYTES * 5])
        with pytest.raises(bank.BankError, match="declares"):
            bank.build({})


class TestTheTransaction:
    def test_t_bank_10_promotion_is_atomic_and_leaves_no_staging(self, kokoro_bundle):
        bank.build({})
        found = bank.promote()
        assert bank.installed()
        assert bank.version() == found["hash"][:16]
        assert not (paths.bank_staging() / paths.BANK_VOICES).exists()

    def test_t_bank_11_a_rollback_restores_the_previous_bytes(self, kokoro_bundle,
                                                              voice_root, voicepack):
        bank.build({})
        bank.promote()
        before = paths.bank_voices().read_bytes()
        bank.build({0: str(voicepack(voice_root / "clone.bin", value=7.0))})
        bank.promote()
        assert paths.bank_voices().read_bytes() != before
        assert bank.rollback() is True
        assert paths.bank_voices().read_bytes() == before

    def test_a_rollback_with_nothing_to_go_back_to_returns_to_the_bundle(self,
                                                                        kokoro_bundle):
        bank.build({})
        bank.promote()
        assert bank.rollback() is False
        assert not bank.installed()
        assert bank.live_paths() == {}

    def test_an_installation_that_never_cloned_anything_uses_the_bundle(self,
                                                                       kokoro_bundle):
        """Section 114. Nobody has to install anything new to keep the speech
        they already had."""
        assert bank.live_paths() == {}
        assert bank.installed() is False

    def test_the_live_paths_point_at_the_bank_once_there_is_one(self, kokoro_bundle):
        bank.build({})
        bank.promote()
        found = bank.live_paths()
        assert found["voices"] == str(paths.bank_voices())
        assert found["model"] == str(paths.bank_model())


class TestStoringAClone:
    def test_a_stored_clone_is_the_normalized_form(self, kokoro_bundle, voice_root,
                                                   voicepack):
        source = voicepack(voice_root / "raw.bin", rows=511)
        stored = bank.store_voicepack("abc123", source)
        assert stored.stat().st_size == SPEAKER_BYTES
        assert stored == paths.clone_file("abc123")

    def test_an_invalid_clone_is_never_stored(self, kokoro_bundle, voice_root, voicepack):
        source = voicepack(voice_root / "raw.bin", rows=99)
        with pytest.raises(bank.BankError):
            bank.store_voicepack("abc123", source)
        assert not paths.clone_file("abc123").exists()

    def test_an_id_that_is_not_a_plain_word_cannot_address_a_path(self):
        """Section 77. The registry is a file on disk, and a corrupted one must
        not be able to name a path outside the root this feature owns."""
        for unsafe in ("../escape", "a/b", "a\\b", ""):
            with pytest.raises(ValueError):
                paths.clone_file(unsafe)


class TestReadingWithoutLoading:
    def test_the_metadata_is_streamed_rather_than_read_whole(self, kokoro_bundle,
                                                             monkeypatch):
        """A Kokoro model is about 350 MB and this is on the path of a status
        route a page polls. Reading the file to count its speakers would make
        polling the status an attack on the user's own machine."""
        def refuse(self, *args, **kwargs):
            raise AssertionError("the whole model file was read into memory")

        monkeypatch.setattr(type(kokoro_bundle), "read_bytes", refuse, raising=False)
        assert bank.speaker_capacity(kokoro_bundle / "model.onnx") == 53

    def test_the_answer_is_cached_and_notices_the_file_changing(self, kokoro_bundle,
                                                                voice_root):
        source = kokoro_bundle / "model.onnx"
        assert bank.speaker_capacity(source) == 53
        bank.derive_model(source, voice_root / "big.onnx", 85)
        (kokoro_bundle / "model.onnx").write_bytes(
            (voice_root / "big.onnx").read_bytes())
        assert bank.speaker_capacity(source) == 85, "the metadata cache went stale"

    def test_a_file_that_is_not_there_answers_nothing_rather_than_raising(self,
                                                                         voice_root):
        assert bank.read_metadata(voice_root / "absent.onnx") == {}
        assert bank.speaker_capacity(voice_root / "absent.onnx") == 0

    def test_a_truncated_model_is_refused_rather_than_half_read(self, kokoro_bundle,
                                                               voice_root):
        broken = voice_root / "broken.onnx"
        broken.write_bytes((kokoro_bundle / "model.onnx").read_bytes()[:12])
        with pytest.raises(bank.BankError):
            bank.derive_model(broken, voice_root / "out.onnx", 85)
