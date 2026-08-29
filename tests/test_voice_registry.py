"""Which voices exist, what they are called, and which number each one is.

The bug this module was written to make impossible is already in the
repository's history: the V1 manifest stored ``voice: af_heart`` beside
``speaker_id: 0``, and in the upstream sherpa Kokoro map speaker 0 is
``af_alloy``. Every reply the feature had ever spoken was spoken by Alloy. The
tests below are mostly about the three things that used to be one thing --
a display name, a stable id, and a number -- staying separate.
"""

from __future__ import annotations

import pytest

import mc_voice_bank as bank
import mc_voice_paths as paths


class TestOfficialVoices:
    def test_t_reg_1_the_pinned_map_puts_heart_at_three(self, voice_registry, kokoro_bundle):
        """Section 42 and section 113: the correction, asserted."""
        found = {entry["id"]: entry["sid"] for entry in voice_registry.official()}
        assert found["official:af_heart"] == 3
        assert found["official:af_alloy"] == 0
        assert found["official:bf_emma"] == 21
        assert found["official:bm_lewis"] == 27

    def test_t_reg_2_every_installed_english_voice_is_offered(self, voice_registry,
                                                             kokoro_bundle):
        found = voice_registry.official()
        american = [entry for entry in found if entry["language"] == "en-US"]
        british = [entry for entry in found if entry["language"] == "en-GB"]
        assert len(american) == 20
        assert len(british) == 8
        assert all(entry["official"] and not entry["editable"] and not entry["deletable"]
                   for entry in found)

    def test_non_english_voices_are_hidden_and_keep_their_positions(self, voice_registry,
                                                                    kokoro_bundle):
        """Section 43: not offered by an English-only release, and their bank
        positions are not destroyed or reordered."""
        shown = {entry["id"] for entry in voice_registry.official()}
        assert "official:zf_xiaobei" not in shown
        assert bank.custom_base() == 53, "hiding a voice changed where custom voices start"

    def test_a_name_is_shown_without_its_accent_prefix(self, voice_registry, kokoro_bundle):
        found = {entry["id"]: entry["display_name"] for entry in voice_registry.official()}
        assert found["official:af_heart"] == "Heart"
        assert found["official:bm_george"] == "George"

    def test_a_bundle_whose_count_disagrees_shows_nothing_rather_than_guessing(
            self, voice_registry, kokoro_bundle, monkeypatch):
        """Names against numbers that might not be theirs is how somebody
        selects Heart and hears Michael."""
        monkeypatch.setattr(bank, "official_names", lambda count=0: [])
        assert voice_registry.official() == []
        assert any("name list" in warning for warning in voice_registry.warnings())


class TestTheDefault:
    def test_a_fresh_installation_starts_at_heart(self, voice_registry, kokoro_bundle):
        assert voice_registry.default_id() == "official:af_heart"
        assert voice_registry.resolve()[0] == 3

    def test_t_reg_8_a_missing_default_falls_back_safely_and_says_so(self, voice_registry,
                                                                    kokoro_bundle):
        voice_registry._remember(voice_registry.OPT_VOICE, "clone:gone")
        assert voice_registry.default_id() == "official:af_heart"
        assert any("no longer installed" in warning for warning in voice_registry.warnings())

    def test_selecting_is_not_setting(self, voice_registry, kokoro_bundle):
        """Section 44: only an explicit Set as Default commits, which is what
        stops somebody auditioning six voices from keeping the last one."""
        voice_registry.resolve("official:bf_emma")
        assert voice_registry.default_id() == "official:af_heart"

    def test_an_unknown_voice_resolves_to_the_default_rather_than_a_number(self,
                                                                          voice_registry,
                                                                          kokoro_bundle):
        """Section 56. A browser-supplied id that means nothing must not become
        a speaker id that means something."""
        sid, entry = voice_registry.resolve("clone:not-a-real-voice")
        assert entry["id"] == voice_registry.default_id()
        assert sid == 3

    def test_setting_an_unknown_default_is_refused(self, voice_registry, kokoro_bundle):
        with pytest.raises(voice_registry.RegistryError):
            voice_registry.set_default("official:not_a_voice")


class TestCustomVoices:
    def test_a_clone_is_registered_at_the_first_reserved_slot(self, voice_registry,
                                                              kokoro_bundle, voice_root,
                                                              voicepack):
        source = voicepack(voice_root / "a.bin")
        entry = voice_registry.register("Alice", "en-US", source)
        assert entry["slot"] == 0
        assert entry["sid"] == 53
        assert entry["id"].startswith("clone:")

    def test_t_reg_3_the_asterisk_is_display_only(self, voice_registry, kokoro_bundle,
                                                  voice_root, voicepack):
        """Section 41. Not in the stored name, not in the id, not in a filename
        -- so a clone called "Alice" beside an official "Alice" is a display
        question and nothing more."""
        entry = voice_registry.register("Alice", "en-US", voicepack(voice_root / "a.bin"))
        assert entry["label"] == "* Alice"
        assert entry["display_name"] == "Alice"
        assert "*" not in entry["id"]
        stored = paths.registry_path().read_text(encoding="utf-8")
        assert '"* Alice"' not in stored

    def test_a_clone_is_proved_through_the_production_runtime_before_it_counts(
            self, voice_registry, kokoro_bundle, voice_root, voicepack):
        """Section 55, step 10, and the reason it is not ceremony: sherpa answers
        an out-of-range speaker by using speaker 0, so a bank whose metadata did
        not take would produce a clone that works and sounds like somebody
        else."""
        entry = voice_registry.register("Alice", "en-US", voicepack(voice_root / "a.bin"))
        assert voice_registry.spoken[-1] == entry["sid"]

    def test_a_clone_whose_check_fails_registers_nothing(self, voice_registry,
                                                         kokoro_bundle, voice_root,
                                                         voicepack, monkeypatch):
        import mc_voice_runtime

        monkeypatch.setattr(mc_voice_runtime, "synthesize",
                            lambda text, sid=0, speed=1.0: b"")
        with pytest.raises(voice_registry.RegistryError, match="did not pass"):
            voice_registry.register("Alice", "en-US", voicepack(voice_root / "a.bin"))
        assert voice_registry.custom() == []
        assert not bank.installed(), "a bank that failed its check was left installed"

    def test_t_reg_4_rename_changes_nothing_but_the_name(self, voice_registry,
                                                         kokoro_bundle, voice_root,
                                                         voicepack):
        entry = voice_registry.register("Alice", "en-US", voicepack(voice_root / "a.bin"))
        before = dict(entry)
        renamed = voice_registry.rename(entry["id"], "Alicia")
        assert renamed["display_name"] == "Alicia"
        for key in ("id", "slot", "sid", "type", "language"):
            assert renamed[key] == before[key], key
        assert paths.clone_file(entry["id"].split(":", 1)[1]).is_file()

    def test_t_reg_5_official_voices_cannot_be_renamed_or_deleted(self, voice_registry,
                                                                  kokoro_bundle):
        with pytest.raises(voice_registry.RegistryError, match="cannot be renamed"):
            voice_registry.rename("official:af_heart", "Mine")
        with pytest.raises(voice_registry.RegistryError, match="cannot be deleted"):
            voice_registry.delete("official:af_heart")

    def test_t_reg_6_a_clone_can_be_the_default(self, voice_registry, kokoro_bundle,
                                               voice_root, voicepack):
        entry = voice_registry.register("Alice", "en-US", voicepack(voice_root / "a.bin"))
        voice_registry.set_default(entry["id"])
        assert voice_registry.resolve()[0] == entry["sid"]

    def test_t_reg_7_deleting_the_default_switches_it_first(self, voice_registry,
                                                            kokoro_bundle, voice_root,
                                                            voicepack):
        """Section 47. The order is the design: a deleted voice must never be
        the one the next reply resolves to."""
        entry = voice_registry.register("Alice", "en-US", voicepack(voice_root / "a.bin"))
        voice_registry.set_default(entry["id"])
        voice_registry.delete(entry["id"])
        assert voice_registry.default_id() == "official:af_heart"
        assert voice_registry.custom() == []

    def test_t_bank_9_deleting_one_clone_leaves_the_others_where_they_were(
            self, voice_registry, kokoro_bundle, voice_root, voicepack):
        """Release blocker seven. Renumber slot 3 down into slot 2 and everybody
        who had selected "Bob" is now listening to "Carol"."""
        first = voice_registry.register("Alice", "en-US", voicepack(voice_root / "a.bin"))
        second = voice_registry.register("Bob", "en-GB", voicepack(voice_root / "b.bin"))
        third = voice_registry.register("Carol", "en-US", voicepack(voice_root / "c.bin"))
        voice_registry.delete(second["id"])
        assert voice_registry.lookup(first["id"])["sid"] == first["sid"]
        assert voice_registry.lookup(third["id"])["sid"] == third["sid"]

    def test_a_freed_slot_is_reused_by_the_next_clone(self, voice_registry, kokoro_bundle,
                                                      voice_root, voicepack):
        first = voice_registry.register("Alice", "en-US", voicepack(voice_root / "a.bin"))
        second = voice_registry.register("Bob", "en-GB", voicepack(voice_root / "b.bin"))
        voice_registry.delete(first["id"])
        third = voice_registry.register("Carol", "en-US", voicepack(voice_root / "c.bin"))
        assert third["slot"] == 0
        assert voice_registry.lookup(second["id"])["sid"] == second["sid"]

    def test_deleting_removes_the_canonical_file_only_after_the_commit(self,
                                                                      voice_registry,
                                                                      kokoro_bundle,
                                                                      voice_root,
                                                                      voicepack):
        entry = voice_registry.register("Alice", "en-US", voicepack(voice_root / "a.bin"))
        source = paths.clone_file(entry["id"].split(":", 1)[1])
        assert source.is_file()
        voice_registry.delete(entry["id"])
        assert not source.exists()

    def test_a_missing_clone_file_is_reported_rather_than_hidden(self, voice_registry,
                                                                 kokoro_bundle,
                                                                 voice_root, voicepack):
        entry = voice_registry.register("Alice", "en-US", voicepack(voice_root / "a.bin"))
        paths.clone_file(entry["id"].split(":", 1)[1]).unlink()
        assert any("missing its voice file" in warning
                   for warning in voice_registry.warnings())


class TestNamesAndCapacity:
    def test_a_name_is_a_name_and_cannot_be_anything_else(self, voice_registry):
        assert voice_registry.check_name("  Alice O'Hara-Smith (2)  ") == "Alice O'Hara-Smith (2)"
        for bad in ("", "   ", "x" * 200, "../etc/passwd", "a; rm -rf /", "$(whoami)"):
            with pytest.raises(voice_registry.RegistryError):
                voice_registry.check_name(bad)

    def test_t_api_12_a_full_bank_refuses_before_anything_expensive_starts(
            self, voice_registry, kokoro_bundle, voice_root, voicepack, monkeypatch):
        """Section 76: refused before hours of optimization, not after."""
        monkeypatch.setattr(bank, "CUSTOM_CAPACITY", 1)
        voice_registry.register("Alice", "en-US", voicepack(voice_root / "a.bin"))
        assert voice_registry.capacity() == {"used": 1, "total": 1, "free": 0}
        with pytest.raises(voice_registry.RegistryError, match="capacity is full"):
            voice_registry.free_slot()

    def test_the_highest_sid_is_what_the_handshake_checks_against(self, voice_registry,
                                                                  kokoro_bundle,
                                                                  voice_root, voicepack):
        assert voice_registry.highest_sid() is None
        entry = voice_registry.register("Alice", "en-US", voicepack(voice_root / "a.bin"))
        assert voice_registry.highest_sid() == entry["sid"]


class TestTheTestText:
    def test_it_has_the_documented_default_and_is_bounded(self, voice_registry):
        assert voice_registry.test_text() == "This is a test of voice cloning."
        assert voice_registry.set_test_text("x" * 5000) == "x" * voice_registry.MAX_TEST_CHARS
        assert voice_registry.set_test_text("   ") == voice_registry.DEFAULT_TEST_TEXT
