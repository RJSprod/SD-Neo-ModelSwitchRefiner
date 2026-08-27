"""The curated catalogue, and the rule that it is the only thing reachable.

The registry is the trust root for a feature that downloads gigabytes and then
runs them, so the tests that matter most here are the refusals. Every one of
them is against a file inside the extension and therefore against a mistake a
reviewer would have to make, which is exactly why they are worth having: the
cost of catching a traversable filename at load time is nothing, and the cost
of catching it at write time is somebody's home directory.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import mc_llm_managed_models as managed
import mc_llm_paths


@pytest.fixture(autouse=True)
def root(tmp_path, monkeypatch, host):
    """A throwaway LLM data root, and a registry cache that does not leak."""
    install = tmp_path / "install"
    install.mkdir()
    monkeypatch.setattr(mc_llm_paths, "data_root", lambda: install)
    managed._registry_cache = None
    yield install
    managed._registry_cache = None


def write_registry(tmp_path, monkeypatch, document) -> Path:
    """Point the module at a registry we wrote, for the refusal cases."""
    path = tmp_path / "registry.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    monkeypatch.setattr(managed, "REGISTRY_PATH", path)
    managed._registry_cache = None
    return path


def entry_document(**overrides) -> dict:
    """One well-formed row, with whatever is being tested replaced."""
    row = {
        "id": "test-model",
        "label": "Test Model",
        "role": "Recommended",
        "group": "Recommended",
        "family": "Test",
        "profile": "gemma4-12b-qat-balanced",
        "multimodal": True,
        "source_url": "https://huggingface.co/example/test",
        "repo_id": "example/test",
        "revision": "main",
        "model": {"filename": "test-Q4_K_M.gguf", "sha256": "a" * 64, "bytes": None,
                  "display_size": "~7.4 GB"},
        "projector": {"filename": "mmproj-test.gguf", "sha256": "b" * 64, "bytes": None,
                      "display_size": "175 MB"},
    }
    row.update(overrides)
    return {"version": 1, "registry_version": "test-1", "models": [row]}


class TestTheShippedCatalogue:
    def test_every_shipped_backbone_is_present_and_in_order(self):
        """Six from the original design intent, plus two groups of three.

        The order is asserted because it is the order Setup draws, and each
        group's own design intent asks for one: quality, then the recommended
        balance, then the smallest. The Q4_K_M 26B entry stays last of all.
        Losing it would remove the automated route back to a comparison, which
        is the whole reason any newer tier can be trusted at all.
        """
        found = managed.catalogue()

        assert [model.identifier for model in found] == [
            "gemma4-12b-qat-balanced",
            "gemma4-e4b-aggressive",
            "qwen35-9b-aggressive",
            "qwen35-9b-defiant-fable",
            "qwen35-4b-aggressive",
            "gemma4-26b-a4b-balanced-q4kp",
            "gemma4-26b-a4b-balanced-q3kp",
            "gemma4-26b-a4b-balanced-q2kp",
            "qwen38-27b-abliterated-q6k",
            "qwen38-27b-abliterated-q5km",
            "qwen38-27b-abliterated-q4km",
            "gemma4-26b-a4b-balanced",
        ]

    def test_every_artifact_carries_a_full_sha256(self):
        """The one field with no fallback. A size can be missing and a revision
        can be a branch; neither can install the wrong bytes, because this is
        what every downloaded byte is checked against."""
        for model in managed.catalogue():
            for artifact in model.artifacts:
                assert len(artifact.sha256) == 64
                assert artifact.sha256 == artifact.sha256.casefold()
                int(artifact.sha256, 16)

    def test_every_entry_names_a_profile_that_exists(self):
        from prompt_master.models import managed_profiles

        for model in managed.catalogue():
            assert managed_profiles.profile(model.profile_id) is not None

    def test_every_entry_shows_where_it_came_from(self):
        """Somebody else's weights under somebody else's licence. A catalogue
        that downloads eight gigabytes without saying whose is not one to ship."""
        for model in managed.catalogue():
            assert model.source_url.startswith("https://huggingface.co/")
            assert model.license_url.startswith("https://")

    def test_every_multimodal_entry_ships_its_own_projector(self):
        for model in managed.catalogue():
            assert model.multimodal
            assert model.projector is not None

    def test_the_creative_qwen_takes_the_regular_build_and_not_the_mtp_one(self):
        """The one entry in the catalogue with a near-identically named trap
        beside it in the same repository."""
        model = managed.entry("qwen35-9b-defiant-fable")

        assert "MTP" not in model.model.filename
        assert model.model.filename.endswith("-Q5_K_M.gguf")

    def test_download_urls_are_built_from_the_pinned_revision(self):
        model = managed.entry("gemma4-12b-qat-balanced")

        assert model.model.url(model.repo_id, model.revision) == (
            "https://huggingface.co/HauhauCS/Gemma4-12B-QAT-Uncensored-HauhauCS-Balanced"
            f"/resolve/{model.revision}/{model.model.filename}")

    def test_the_catalogue_describes_a_choice_and_not_a_settings_page(self):
        """Section 10: role, size, family. Nothing about samplers, cache types
        or templates reaches a user's screen from here."""
        described = managed.entry("gemma4-12b-qat-balanced").describe()

        assert described == "Recommended · ~7.4 GB + 175 MB vision · Gemma 4"
        assert not any(word in described.casefold()
                       for word in ("temperature", "top_k", "top-k", "min_p", "q8_0",
                                    "jinja", "penalty"))


class TestTheGemma26BQuantTiers:
    """Three weights of one backbone, and the one already-shipped entry beside them.

    What makes this worth its own class is that the four are *the same model*.
    Everything that differs between them is a number in a filename, and every
    place the catalogue treats them as four unrelated downloads -- four
    projectors, four profiles that drifted apart, a fifth revision -- is a place
    a user pays for the difference without getting one.
    """

    TIERS = ("gemma4-26b-a4b-balanced-q4kp", "gemma4-26b-a4b-balanced-q3kp",
             "gemma4-26b-a4b-balanced-q2kp")

    REVISION = "96c11c22b1128c3c8c655b21557b409f307c557f"

    def test_each_tier_names_its_own_quantisation(self):
        wanted = {"gemma4-26b-a4b-balanced-q4kp": "Q4_K_P",
                  "gemma4-26b-a4b-balanced-q3kp": "Q3_K_P",
                  "gemma4-26b-a4b-balanced-q2kp": "Q2_K_P"}

        for identifier, quant in wanted.items():
            model = managed.entry(identifier)
            assert model.model.filename.endswith(f"-{quant}.gguf")
            assert "QAT" not in model.model.filename

    def test_they_are_pinned_to_an_immutable_revision(self):
        """Section 3 of the design intent. A branch would still refuse the wrong
        bytes -- the hash does that -- but it would refuse them as a hash
        mismatch, where a pin simply keeps working."""
        for identifier in self.TIERS:
            model = managed.entry(identifier)
            assert model.revision == self.REVISION
            assert model.pinned
            assert model.model.url(model.repo_id, model.revision).endswith(
                f"/resolve/{self.REVISION}/{model.model.filename}")

    def test_they_share_one_vision_projector(self):
        """The same file, byte for byte, in all four. That is what lets the
        downloader link the second and later copies instead of fetching them."""
        entries = [managed.entry(name) for name in
                   self.TIERS + ("gemma4-26b-a4b-balanced",)]
        projectors = {(model.projector.filename, model.projector.sha256)
                      for model in entries}

        assert len(projectors) == 1

    def test_the_shipped_q4_k_m_entry_is_untouched(self):
        """Backward compatibility, and it is a real installation on somebody's
        disk rather than a principle: the id is what their state file names, and
        the hashes are what says the bundle they have is still the right one."""
        model = managed.entry("gemma4-26b-a4b-balanced")

        assert model.model.filename.endswith("-Q4_K_M.gguf")
        assert model.model.sha256 == (
            "f8b1da6dc139e6928159e536bc85602adbc1412018871732a878dedcad7ccafd")
        assert model.profile_id == "gemma4-26b-a4b-balanced"

    def test_each_tier_gets_its_own_hidden_profile(self):
        """Not one profile shared across three quantisations. The balanced and
        low-memory tiers buy their cache back with q8_0 and the quality tier
        does not, which is the difference between them and the reason each has
        its own id in the state file."""
        profiles = {managed.entry(name).profile_id for name in self.TIERS}

        assert profiles == {"gemma4-26b-a4b-q4kp", "gemma4-26b-a4b-q3kp",
                            "gemma4-26b-a4b-q2kp"}

    def test_the_low_memory_tier_is_not_sold_as_an_equal(self):
        """Section 4's last line. Q2 is the lowest-memory option and the
        catalogue may not imply it is the same model at a smaller size."""
        described = managed.entry("gemma4-26b-a4b-balanced-q2kp").describe()

        assert "Low memory" in described
        assert "Recommended" not in described

    def test_the_recommended_tier_is_the_middle_one(self):
        assert "Recommended" in managed.entry("gemma4-26b-a4b-balanced-q3kp").role
        assert "Recommended" not in managed.entry("gemma4-26b-a4b-balanced-q4kp").role

    def test_the_registry_version_moved_with_them(self):
        """A catalogue that gained three entries and kept its version number is
        a catalogue no staged download can tell it has to restart against."""
        assert managed.entry("gemma4-26b-a4b-balanced-q3kp").registry_version != "2026.08.20-1"

    def test_no_mtp_artifact_reaches_these_entries(self):
        """Section 10: the MTP head is published for a different, QAT target and
        must not be attached to these files on the strength of the family name."""
        for identifier in self.TIERS:
            model = managed.entry(identifier)
            assert "MTP" not in model.model.filename.upper()
            assert "MTP" not in model.repo_id.upper()
            assert "QAT" not in model.repo_id.upper()


class TestRefusals:
    def test_an_id_that_is_not_in_the_registry_is_refused(self):
        with pytest.raises(managed.ManagedError):
            managed.entry("something-nobody-shipped")

    @pytest.mark.parametrize("identifier", ["../escape", "a/b", "with space", "UPPER",
                                            "", ".", "a\\b"])
    def test_an_id_that_could_be_a_path_is_refused(self, tmp_path, monkeypatch, identifier):
        """The id becomes a directory name, so this regex is the whole traversal
        defence for the managed root."""
        write_registry(tmp_path, monkeypatch, entry_document(id=identifier))

        with pytest.raises(managed.ManagedError):
            managed.registry()

    @pytest.mark.parametrize("filename", ["../model.gguf", "sub/model.gguf", "model.bin",
                                          "", "model.gguf.exe"])
    def test_a_filename_that_is_not_a_plain_gguf_is_refused(self, tmp_path, monkeypatch,
                                                            filename):
        write_registry(tmp_path, monkeypatch,
                       entry_document(model={"filename": filename, "sha256": "a" * 64}))

        with pytest.raises(managed.ManagedError):
            managed.registry()

    def test_an_incomplete_hash_is_refused(self, tmp_path, monkeypatch):
        write_registry(tmp_path, monkeypatch,
                       entry_document(model={"filename": "m.gguf", "sha256": "a" * 40}))

        with pytest.raises(managed.ManagedError):
            managed.registry()

    def test_a_non_https_source_is_refused(self, tmp_path, monkeypatch):
        write_registry(tmp_path, monkeypatch,
                       entry_document(source_url="http://huggingface.co/example/test"))

        with pytest.raises(managed.ManagedError):
            managed.registry()

    def test_a_moving_latest_revision_is_refused(self, tmp_path, monkeypatch):
        """Section 7.3: never silently update a managed model. A branch is
        allowed and checked by hash; a tag literally called ``latest`` is the
        one name that means "whatever is newest" and is refused outright."""
        write_registry(tmp_path, monkeypatch, entry_document(revision="latest"))

        with pytest.raises(managed.ManagedError):
            managed.registry()

    def test_a_multimodal_entry_with_no_projector_is_refused(self, tmp_path, monkeypatch):
        document = entry_document(multimodal=True)
        document["models"][0].pop("projector")
        write_registry(tmp_path, monkeypatch, document)

        with pytest.raises(managed.ManagedError):
            managed.registry()

    def test_two_entries_with_one_id_are_refused(self, tmp_path, monkeypatch):
        document = entry_document()
        document["models"].append(dict(document["models"][0]))
        write_registry(tmp_path, monkeypatch, document)

        with pytest.raises(managed.ManagedError):
            managed.registry()

    def test_a_missing_registry_is_a_sentence_and_not_a_traceback(self, tmp_path, monkeypatch):
        monkeypatch.setattr(managed, "REGISTRY_PATH", tmp_path / "gone.json")
        managed._registry_cache = None

        with pytest.raises(managed.ManagedError) as raised:
            managed.registry()
        assert "reinstall" in str(raised.value)
        # And the panel's own route never raises at all.
        assert managed.catalogue() == []


class TestSizes:
    @pytest.mark.parametrize("text,expected", [
        ("~7.4 GB", int(7.4 * 1024**3)),
        ("175 MB", 175 * 1024**2),
        ("1.19 GB", int(1.19 * 1024**3)),
        ("nonsense", 0),
        ("", 0),
    ])
    def test_the_display_size_is_read_back_for_planning(self, text, expected):
        """Used for the disk-space check and the progress fraction, and for
        nothing that decides whether a file is the right file."""
        artifact = managed.Artifact("m.gguf", "a" * 64, "model.gguf", None, text)

        assert artifact.approximate_bytes == expected

    def test_an_exact_byte_count_wins_over_the_display_figure(self):
        artifact = managed.Artifact("m.gguf", "a" * 64, "model.gguf", 12345, "~7.4 GB")

        assert artifact.approximate_bytes == 12345


class TestPinning:
    def test_a_commit_revision_reads_as_pinned(self, tmp_path, monkeypatch):
        write_registry(tmp_path, monkeypatch, entry_document(revision="0" * 40))

        assert managed.entry("test-model").pinned

    def test_a_branch_revision_reads_as_unpinned_and_says_so(self, tmp_path, monkeypatch):
        """Unpinned is not unsafe -- the hash still gates every byte -- so the
        catalogue explains the difference rather than hiding the entry."""
        write_registry(tmp_path, monkeypatch, entry_document(revision="main"))
        model = managed.entry("test-model")

        assert not model.pinned
        assert "hash" in managed.status(model).detail


def load_pin_tool():
    """``tools/pin_managed_models.py``, loaded by path.

    By path because ``tools/`` is deliberately not importable: Forge imports
    everything in ``scripts/``, and a maintainer's command-line tool has no
    business being loaded when somebody opens a WebUI.
    """
    import importlib.util
    import sys

    if "pin_managed_models" in sys.modules:
        return sys.modules["pin_managed_models"]
    path = Path(__file__).resolve().parent.parent / "tools" / "pin_managed_models.py"
    spec = importlib.util.spec_from_file_location("pin_managed_models", path)
    module = importlib.util.module_from_spec(spec)
    # Registered before it is executed: ``dataclasses`` resolves a field's
    # annotations through ``sys.modules[cls.__module__]``, so a module that has
    # been executed but never registered cannot declare one.
    sys.modules["pin_managed_models"] = module
    spec.loader.exec_module(module)
    return module


class TestThePinningTool:
    """The tool that turns ``main`` into a commit. Its refusal is the point."""

    def document(self):
        return {"models": [{
            "id": "test-model", "repo_id": "example/test", "revision": "main",
            "model": {"filename": "weights.gguf", "sha256": "a" * 64, "bytes": None},
            "projector": {"filename": "mmproj.gguf", "sha256": "b" * 64, "bytes": None},
        }]}

    def resolver(self, commit="c" * 40, files=None):
        tool = load_pin_tool()
        answer = tool.Resolved(commit, files if files is not None else {
            "weights.gguf": (7_000_000, "a" * 64),
            "mmproj.gguf": (175_000, "b" * 64)})
        return lambda *_args: answer

    def test_it_writes_the_commit_and_the_exact_byte_counts(self):
        tool = load_pin_tool()

        updated, changes = tool.pin(self.document(), self.resolver())

        assert updated["models"][0]["revision"] == "c" * 40
        assert updated["models"][0]["model"]["bytes"] == 7_000_000
        assert updated["models"][0]["projector"]["bytes"] == 175_000
        assert len(changes) == 3

    def test_it_pins_the_files_an_accelerator_names_too(self):
        """The DFlash2 draft is a file in the same repository at the same
        revision. Leaving it out would leave one artifact in the catalogue
        whose committed hash nobody had compared against the hub."""
        tool = load_pin_tool()
        document = self.document()
        document["models"][0]["accelerators"] = {
            "mtp": {"embedded": True, "draft_tokens": 3},
            "dflash2": {"filename": "draft.gguf", "sha256": "d" * 64, "bytes": None},
        }
        resolver = self.resolver(files={
            "weights.gguf": (7_000_000, "a" * 64),
            "mmproj.gguf": (175_000, "b" * 64),
            "draft.gguf": (3_860_000_000, "d" * 64)})

        updated, changes = tool.pin(document, resolver)

        assert updated["models"][0]["accelerators"]["dflash2"]["bytes"] == 3_860_000_000
        assert any("draft.gguf" in line for line in changes)

    def test_a_draft_whose_hash_moved_refuses_the_whole_run(self):
        tool = load_pin_tool()
        document = self.document()
        document["models"][0]["accelerators"] = {
            "dflash2": {"filename": "draft.gguf", "sha256": "d" * 64, "bytes": None}}
        resolver = self.resolver(files={
            "weights.gguf": (7_000_000, "a" * 64),
            "mmproj.gguf": (175_000, "b" * 64),
            "draft.gguf": (3_860_000_000, "e" * 64)})

        with pytest.raises(tool.PinError, match="draft.gguf"):
            tool.pin(document, resolver)

    def test_it_never_writes_the_hubs_hash_over_the_committed_one(self):
        """The checked-in SHA-256 is the trust root for the whole feature. A
        publisher whose files have really changed is a review decision."""
        tool = load_pin_tool()
        resolver = self.resolver(files={"weights.gguf": (7_000_000, "d" * 64),
                                        "mmproj.gguf": (175_000, "b" * 64)})

        with pytest.raises(tool.PinError) as raised:
            tool.pin(self.document(), resolver)

        assert "Nothing was written" in str(raised.value)

    def test_a_hub_that_reports_no_hash_still_gets_the_size_pinned(self):
        """The download verifies against what is checked in either way, so a
        missing etag costs precision in a progress bar and nothing more."""
        tool = load_pin_tool()
        resolver = self.resolver(files={"weights.gguf": (7_000_000, ""),
                                        "mmproj.gguf": (175_000, "")})

        updated, _changes = tool.pin(self.document(), resolver)

        assert updated["models"][0]["model"]["bytes"] == 7_000_000

    def test_it_reads_the_size_of_the_lfs_object_and_not_of_the_pointer(self):
        """A plain content-length on an LFS pointer is a few hundred bytes: a
        number that looks like an answer and is not one."""
        tool = load_pin_tool()

        assert tool._size({"Content-Length": "134", "X-Linked-Size": "7000000"}) == 7_000_000

    @pytest.mark.parametrize("headers,expected", [
        ({"ETag": '"' + "a" * 64 + '"'}, "a" * 64),
        ({"X-Linked-Etag": "sha256:" + "B" * 64}, "b" * 64),
        ({"ETag": '"W/abc123"'}, ""),
        ({}, ""),
    ])
    def test_only_something_that_really_is_a_sha256_is_read_as_one(self, headers, expected):
        tool = load_pin_tool()

        assert tool._sha256(headers) == expected

    def test_a_revision_answer_without_a_commit_is_refused(self):
        tool = load_pin_tool()

        with pytest.raises(tool.PinError):
            tool.resolve("example/test", "main", ["weights.gguf"],
                         lambda _method, _url: (200, {}, json.dumps({"siblings": []})))
