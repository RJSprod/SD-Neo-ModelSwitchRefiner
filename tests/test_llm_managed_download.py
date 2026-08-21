"""Downloading a managed bundle, and the failures that must cost nothing.

The interesting assertions in this file are all the same assertion said six
ways: *after this went wrong, is the model the user was running still the model
they are running?* A download is the one operation in LLM Studio that takes
twenty minutes and touches gigabytes, so it is also the one where "it failed and
now the install is broken" would be worst -- and the design that prevents it
(stage, verify, then one rename) is only worth anything if every failure path
really does stop before the rename.

The HTTP layer is faked at ``downloader._client`` rather than higher up on
purpose: the resume, the 416-means-start-again rule, the SHA-256 check and the
verify-then-rename all live in the vendored downloader, and a test that mocked
past them would be testing the fake.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import httpx
import pytest

import mc_llm_managed_models as managed
import mc_llm_paths
from test_llm_context import build_model


# --------------------------------------------------------------------------- #
# A Hugging Face that is not one
# --------------------------------------------------------------------------- #


class _Response:
    def __init__(self, url, payload, status_code):
        self._payload, self.status_code, self._url = payload, status_code, url

    def raise_for_status(self):
        if self.status_code < 400:
            return
        raise httpx.HTTPStatusError(
            f"{self.status_code}", request=httpx.Request("GET", self._url),
            response=httpx.Response(self.status_code))

    def iter_bytes(self, size):
        for start in range(0, len(self._payload), max(size, 1)):
            yield self._payload[start:start + size]

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False


class FakeHub:
    """Serves exactly the filenames it was given, with HTTP Range support."""

    def __init__(self, files: dict[str, bytes]):
        self.files = files
        self.requests: list[tuple[str, dict]] = []
        self.fail_next = 0

    # ``downloader._client()`` returns this, and it is used as a context manager.
    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False

    def stream(self, _method, url, headers=None):
        headers = dict(headers or {})
        self.requests.append((url, headers))
        body = self.files.get(url.rsplit("/", 1)[-1])
        if body is None:
            return _Response(url, b"", 404)
        if self.fail_next:
            self.fail_next -= 1
            return _Response(url, b"", 503)
        offset = 0
        if headers.get("Range"):
            offset = int(headers["Range"].split("=", 1)[1].split("-", 1)[0])
            if offset >= len(body):
                return _Response(url, b"", 416)
            return _Response(url, body[offset:], 206)
        return _Response(url, body, 200)


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture(autouse=True)
def root(tmp_path, monkeypatch, host):
    install = tmp_path / "install"
    (install / "data").mkdir(parents=True)
    monkeypatch.setattr(mc_llm_paths, "data_root", lambda: install)
    monkeypatch.setattr("prompt_master.provisioning.downloader.time.sleep", lambda _s: None)
    managed._registry_cache = None
    yield install
    managed._registry_cache = None


@pytest.fixture
def artifacts(tmp_path):
    """Two real GGUFs, small enough to hash in a test."""
    source = tmp_path / "source"
    source.mkdir(parents=True, exist_ok=True)
    model = build_model(source, "weights.gguf", size_mb=1).read_bytes()
    mmproj = build_model(source, "projector.gguf", size_mb=1, blocks=4).read_bytes()
    return {"weights.gguf": model, "mmproj-test.gguf": mmproj}


@pytest.fixture
def hub(monkeypatch, artifacts):
    served = FakeHub(dict(artifacts))
    monkeypatch.setattr("prompt_master.provisioning.downloader._client", lambda: served)
    return served


@pytest.fixture
def registry(tmp_path, monkeypatch, artifacts):
    """A one-entry catalogue whose hashes are the real hashes of those files."""
    def entry(name):
        return {"filename": name,
                "sha256": hashlib.sha256(artifacts[name]).hexdigest(),
                "bytes": len(artifacts[name]),
                "display_size": f"{len(artifacts[name])} B"}

    document = {
        "version": 1, "registry_version": "test-1",
        "models": [{
            "id": "test-model", "label": "Test Model", "role": "Recommended",
            "group": "Recommended", "family": "Test",
            "profile": "gemma4-12b-qat-balanced", "multimodal": True,
            "source_url": "https://huggingface.co/example/test",
            "repo_id": "example/test", "revision": "main",
            "model": entry("weights.gguf"), "projector": entry("mmproj-test.gguf"),
        }],
    }
    path = tmp_path / "registry.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    monkeypatch.setattr(managed, "REGISTRY_PATH", path)
    managed._registry_cache = None
    return document


def state_of(root: Path) -> dict:
    path = root / "data" / "setup-state.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def write_state(root: Path, **values) -> None:
    path = root / "data" / "setup-state.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(values), encoding="utf-8")


# --------------------------------------------------------------------------- #


class TestAGoodDownload:
    def test_it_lands_as_a_bundle_with_fixed_filenames(self, root, hub, registry):
        bundle = managed.download("test-model")

        assert bundle.model == root / "models" / "managed" / "test-model" / "model.gguf"
        assert bundle.mmproj == root / "models" / "managed" / "test-model" / "mmproj.gguf"
        assert bundle.model.is_file() and bundle.mmproj.is_file()

    def test_the_publishers_own_filenames_are_recorded_but_never_used_as_paths(
            self, root, hub, registry):
        """A filename out of a JSON file is a string, and the set of strings
        that are safe to join onto a path is smaller than the set that look
        like it. What is on disk is fixed; what it *was* is written down."""
        managed.download("test-model")
        document = json.loads(
            (root / "models" / "managed" / "test-model" / "installed.json").read_text())

        assert document["artifacts"]["model"]["filename"] == "weights.gguf"
        assert document["artifacts"]["model"]["stored_as"] == "model.gguf"
        assert not list((root / "models" / "managed" / "test-model").glob("weights.gguf"))

    def test_it_records_what_it_installed_and_the_profile_it_belongs_to(self, root, hub,
                                                                       registry):
        from prompt_master.models import managed_profiles

        bundle = managed.download("test-model")

        assert bundle.registry_version == "test-1"
        assert bundle.revision == "main"
        assert bundle.profile_id == "gemma4-12b-qat-balanced"
        assert bundle.profile_version == managed_profiles.VERSION

    def test_it_fetches_from_the_registrys_revision_and_nowhere_else(self, hub, registry):
        managed.download("test-model")

        assert [url for url, _headers in hub.requests] == [
            "https://huggingface.co/example/test/resolve/main/weights.gguf",
            "https://huggingface.co/example/test/resolve/main/mmproj-test.gguf",
        ]

    def test_nothing_is_left_in_staging(self, root, hub, registry):
        managed.download("test-model")

        assert not (root / "models" / "managed" / ".downloads" / "test-model").exists()

    def test_the_sidecar_does_not_survive_into_an_installed_bundle(self, root, hub, registry):
        managed.download("test-model")

        assert not (root / "models" / "managed" / "test-model" / "download-state.json").exists()

    def test_it_changes_no_selection_at_all(self, root, hub, registry):
        """Downloaded is not selected (section 6). Somebody who fetches a second
        backbone to try later is still running the first one afterwards."""
        write_state(root, model="/elsewhere/mine.gguf", runtime="runtime/llama-server")

        managed.download("test-model")

        assert state_of(root)["model"] == "/elsewhere/mine.gguf"
        assert "managed_model_id" not in state_of(root)

    def test_a_bundle_that_is_already_here_is_never_downloaded_twice(self, root, hub,
                                                                     registry):
        """The user's own words for this feature. A backbone on disk that still
        matches the catalogue costs no bytes and no staging directory, whichever
        route asked for it."""
        first = managed.download("test-model")
        hub.requests.clear()

        again = managed.download("test-model")

        assert hub.requests == []
        assert again.model == first.model
        assert not (root / "models" / "managed" / ".downloads" / "test-model").exists()


class TestItIsSmartAboutWhatIsAlreadyHere:
    def test_an_installed_bundle_reads_as_installed_without_re_hashing(self, hub, registry):
        managed.download("test-model")
        model = managed.entry("test-model")

        assert managed.status(model).state == managed.INSTALLED
        assert managed.status(model).ready

    def test_a_bundle_the_catalogue_has_moved_on_from_is_flagged_and_kept(
            self, tmp_path, monkeypatch, hub, registry):
        managed.download("test-model")
        registry["models"][0]["model"]["sha256"] = "c" * 64
        (tmp_path / "registry.json").write_text(json.dumps(registry), encoding="utf-8")
        managed._registry_cache = None

        found = managed.status(managed.entry("test-model"))

        assert found.state == managed.SUPERSEDED
        assert not found.ready
        assert found.bundle is not None and found.bundle.model.is_file()

    def test_an_interrupted_download_reads_as_interrupted_and_not_as_installed(
            self, root, registry):
        staging = root / "models" / "managed" / ".downloads" / "test-model"
        staging.mkdir(parents=True)
        (staging / "model.gguf.part").write_bytes(b"half a model")

        found = managed.status(managed.entry("test-model"))

        assert found.state == managed.PARTIAL
        assert not found.ready
        assert managed.installed("test-model") is None

    def test_a_download_resumes_from_what_is_already_on_disk(self, root, hub, registry,
                                                             artifacts):
        """Only with a sidecar that matches: see the next test for why."""
        managed._prepare_staging(managed.entry("test-model"),
                                 managed.staging_root("test-model"))
        staging = root / "models" / "managed" / ".downloads" / "test-model"
        (staging / "model.gguf.part").write_bytes(artifacts["weights.gguf"][:4096])

        managed.download("test-model")

        ranges = [headers.get("Range") for url, headers in hub.requests
                  if url.endswith("weights.gguf")]
        assert ranges == ["bytes=4096-"]
        assert managed.installed("test-model").model.read_bytes() == artifacts["weights.gguf"]

    def test_a_part_file_from_a_different_catalogue_entry_is_discarded(self, root, hub,
                                                                      registry, artifacts):
        """A ``.part`` file is a pile of bytes with no memory of what it was
        going to be. Appending the current model to the previous one's prefix
        produces a file of exactly the right length that is not any model."""
        staging = root / "models" / "managed" / ".downloads" / "test-model"
        staging.mkdir(parents=True)
        (staging / "model.gguf.part").write_bytes(b"bytes from an older quantisation")
        (staging / "download-state.json").write_text(
            json.dumps({"expected": {"revision": "something-else"}}), encoding="utf-8")

        managed.download("test-model")

        assert not any(headers.get("Range") for _url, headers in hub.requests)
        assert managed.installed("test-model").model.read_bytes() == artifacts["weights.gguf"]

    def test_a_part_file_longer_than_the_file_itself_starts_again(self, root, hub, registry,
                                                                  artifacts):
        """A re-uploaded release: every resume from that part file is a 416, so
        the only way out is to start the file over."""
        managed._prepare_staging(managed.entry("test-model"),
                                 managed.staging_root("test-model"))
        staging = root / "models" / "managed" / ".downloads" / "test-model"
        (staging / "model.gguf.part").write_bytes(artifacts["weights.gguf"] + b"extra")

        managed.download("test-model")

        assert managed.installed("test-model").model.read_bytes() == artifacts["weights.gguf"]


class TestOneProjectorSharedBySeveralBackbones:
    """Four Gemma 26B tiers name one vision projector: same file, same hash.

    Downloading it once per tier is 1.19 GB of somebody's connection and 1.19 GB
    of their disk, four times over, for four names of one file. So a bundle that
    already holds exactly those bytes is linked rather than fetched -- and the
    thing being tested is as much the fallback as the optimisation, because a
    disk optimisation that can fail a download is not worth having.
    """

    @pytest.fixture
    def tiers(self, tmp_path, monkeypatch, artifacts, hub):
        """Two entries of one family, sharing a projector byte for byte."""
        second = build_model(tmp_path / "source", "weights-2.gguf", size_mb=1,
                             blocks=8).read_bytes()
        artifacts["weights-2.gguf"] = second
        hub.files["weights-2.gguf"] = second

        def entry(name):
            return {"filename": name,
                    "sha256": hashlib.sha256(artifacts[name]).hexdigest(),
                    "bytes": len(artifacts[name]),
                    "display_size": f"{len(artifacts[name])} B"}

        def row(identifier, weights):
            return {"id": identifier, "label": identifier, "role": "Quality",
                    "group": "Family", "family": "Test",
                    "profile": "gemma4-12b-qat-balanced", "multimodal": True,
                    "source_url": "https://huggingface.co/example/test",
                    "repo_id": "example/test", "revision": "main",
                    "model": entry(weights), "projector": entry("mmproj-test.gguf")}

        document = {"version": 1, "registry_version": "test-1",
                    "models": [row("first-tier", "weights.gguf"),
                               row("second-tier", "weights-2.gguf")]}
        path = tmp_path / "tiers.json"
        path.write_text(json.dumps(document), encoding="utf-8")
        monkeypatch.setattr(managed, "REGISTRY_PATH", path)
        managed._registry_cache = None
        return document

    def test_the_second_tier_does_not_download_the_projector_again(self, hub, tiers):
        managed.download("first-tier")
        hub.requests.clear()

        managed.download("second-tier")

        asked = [url.rsplit("/", 1)[-1] for url, _headers in hub.requests]
        assert asked == ["weights-2.gguf"]

    def test_the_second_tier_still_has_a_projector(self, root, hub, tiers):
        managed.download("first-tier")
        bundle = managed.download("second-tier")

        assert bundle.mmproj is not None
        assert bundle.mmproj.is_file()
        assert bundle.mmproj.read_bytes() == (
            root / "models" / "managed" / "first-tier" / "mmproj.gguf").read_bytes()

    def test_it_is_one_file_on_disk_and_not_two(self, root, hub, tiers):
        """A hard link, where the filesystem allows one: the same inode under
        two names, so the second tier costs nothing at all."""
        managed.download("first-tier")
        managed.download("second-tier")

        first = (root / "models" / "managed" / "first-tier" / "mmproj.gguf").stat()
        second = (root / "models" / "managed" / "second-tier" / "mmproj.gguf").stat()

        assert (first.st_dev, first.st_ino) == (second.st_dev, second.st_ino)

    def test_a_filesystem_that_will_not_link_still_installs(self, root, hub, tiers,
                                                            monkeypatch):
        """Cross-device, or Windows without the privilege. The copy costs the
        disk space the link would have saved and saves the download either way."""
        def refuse(*_args, **_kwargs):
            raise OSError("cross-device link")

        managed.download("first-tier")
        monkeypatch.setattr(managed.os, "link", refuse)
        hub.requests.clear()

        bundle = managed.download("second-tier")

        assert bundle.mmproj.is_file()
        assert [url.rsplit("/", 1)[-1] for url, _h in hub.requests] == ["weights-2.gguf"]
        first = (root / "models" / "managed" / "first-tier" / "mmproj.gguf").stat()
        assert first.st_ino != bundle.mmproj.stat().st_ino

    def test_a_local_copy_that_no_longer_hashes_is_downloaded_instead(self, root, hub,
                                                                     tiers):
        """The hash in the manifest describes what was promoted, which is a
        statement about the past. This is somebody's disk."""
        managed.download("first-tier")
        (root / "models" / "managed" / "first-tier" / "mmproj.gguf").write_bytes(b"tampered")
        hub.requests.clear()

        bundle = managed.download("second-tier")

        asked = [url.rsplit("/", 1)[-1] for url, _headers in hub.requests]
        assert "mmproj-test.gguf" in asked
        assert bundle.mmproj.is_file()

    def test_nothing_is_reused_across_a_different_hash(self, hub, tiers):
        """The model files differ, so only the projector may be shared."""
        managed.download("first-tier")
        hub.requests.clear()

        managed.download("second-tier")

        assert "weights.gguf" not in [url.rsplit("/", 1)[-1] for url, _h in hub.requests]


class TestFailuresCostNothing:
    def test_a_hash_mismatch_installs_nothing_and_says_the_extension_is_stale(
            self, root, hub, registry, tmp_path):
        registry["models"][0]["model"]["sha256"] = "c" * 64
        (tmp_path / "registry.json").write_text(json.dumps(registry), encoding="utf-8")
        managed._registry_cache = None
        write_state(root, model="/elsewhere/mine.gguf")

        with pytest.raises(managed.ManagedError) as raised:
            managed.download("test-model")

        assert "update the extension" in str(raised.value)
        assert not (root / "models" / "managed" / "test-model").exists()
        assert state_of(root)["model"] == "/elsewhere/mine.gguf"

    def test_a_failed_hash_leaves_no_part_file_to_poison_the_next_attempt(
            self, root, hub, registry, tmp_path):
        registry["models"][0]["model"]["sha256"] = "c" * 64
        (tmp_path / "registry.json").write_text(json.dumps(registry), encoding="utf-8")
        managed._registry_cache = None

        with pytest.raises(managed.ManagedError):
            managed.download("test-model")

        staging = root / "models" / "managed" / ".downloads" / "test-model"
        assert not (staging / "model.gguf.part").exists()
        assert not (staging / "model.gguf").exists()

    def test_a_404_fails_closed_rather_than_finding_something_else(self, root, hub, registry):
        hub.files.pop("weights.gguf")

        with pytest.raises(managed.ManagedError):
            managed.download("test-model")

        assert not (root / "models" / "managed" / "test-model").exists()

    def test_a_file_that_hashes_correctly_and_is_not_a_gguf_is_refused(
            self, root, monkeypatch, registry, artifacts, tmp_path):
        """A publisher who uploads the wrong kind of file under a .gguf name
        produces a bundle that verifies and then fails at llama-server startup,
        several minutes and one discarded model later."""
        payload = b"not a gguf at all"
        registry["models"][0]["model"].update(
            {"sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload)})
        (tmp_path / "registry.json").write_text(json.dumps(registry), encoding="utf-8")
        managed._registry_cache = None
        served = FakeHub({"weights.gguf": payload,
                          "mmproj-test.gguf": artifacts["mmproj-test.gguf"]})
        monkeypatch.setattr("prompt_master.provisioning.downloader._client", lambda: served)

        with pytest.raises(managed.ManagedError) as raised:
            managed.download("test-model")

        assert "not a GGUF" in str(raised.value)
        assert not (root / "models" / "managed" / "test-model").exists()

    def test_a_missing_projector_fails_the_whole_bundle(self, root, hub, registry):
        """Section 7.1 step 5: a required projector failing fails the bundle.
        Half a multimodal backbone is not a backbone."""
        hub.files.pop("mmproj-test.gguf")

        with pytest.raises(managed.ManagedError):
            managed.download("test-model")

        assert not (root / "models" / "managed" / "test-model").exists()

    def test_a_full_disk_is_refused_before_anything_is_created(self, root, hub, registry,
                                                              monkeypatch):
        import shutil as shutil_module

        monkeypatch.setattr(managed.shutil, "disk_usage",
                            lambda _path: shutil_module._ntuple_diskusage(100, 100, 0))

        with pytest.raises(managed.ManagedError) as raised:
            managed.download("test-model")

        assert "headroom" in str(raised.value)
        assert hub.requests == []
        assert not (root / "models" / "managed" / ".downloads" / "test-model").exists()

    def test_a_download_that_cannot_be_promoted_puts_the_old_bundle_back(
            self, root, hub, registry, artifacts, tmp_path, monkeypatch):
        """Replacing a bundle is a rename, so it fails whole or not at all.
        Deleting the old one first would mean an install that had a working
        model before the attempt had none after it."""
        managed.download("test-model")
        installed_model = root / "models" / "managed" / "test-model" / "model.gguf"
        first = installed_model.read_bytes()

        # The catalogue moves to a different build of the same backbone, so the
        # bundle on disk no longer matches and a real promotion is attempted.
        replacement = build_model(tmp_path / "source", "weights-v2.gguf", blocks=8,
                                  size_mb=1).read_bytes()
        registry["models"][0]["model"].update(
            {"filename": "weights-v2.gguf", "bytes": len(replacement),
             "sha256": hashlib.sha256(replacement).hexdigest()})
        (tmp_path / "registry.json").write_text(json.dumps(registry), encoding="utf-8")
        managed._registry_cache = None
        hub.files["weights-v2.gguf"] = replacement

        real_rename = Path.rename

        def refuse(self, target):
            # Only the promotion. A fake that also broke the restore would be
            # testing a filesystem that cannot roll back rather than code that
            # does not.
            if self.parent.name == ".downloads":
                raise OSError("in use")
            return real_rename(self, target)

        monkeypatch.setattr(Path, "rename", refuse)
        with pytest.raises(managed.ManagedError) as raised:
            managed.download("test-model")

        assert "nothing about the model you are using has changed" in str(raised.value)
        assert installed_model.read_bytes() == first

    def test_cancelling_keeps_what_arrived_and_changes_no_selection(self, root, hub, registry):
        import threading

        write_state(root, model="/elsewhere/mine.gguf")
        cancel = threading.Event()
        cancel.set()

        with pytest.raises(managed.Cancelled):
            managed.download("test-model", cancel=cancel)

        assert not (root / "models" / "managed" / "test-model").exists()
        assert state_of(root)["model"] == "/elsewhere/mine.gguf"


class TestContainment:
    def test_every_managed_path_resolves_under_the_managed_root(self, root):
        managed_root = root / "models" / "managed"

        assert managed.bundle_root("test-model").parent == managed_root
        assert managed.staging_root("test-model").parent == managed_root / ".downloads"

    @pytest.mark.parametrize("identifier", ["../escape", "a/b", "", "."])
    def test_a_path_shaped_id_never_reaches_the_filesystem(self, identifier):
        with pytest.raises(managed.ManagedError):
            managed.bundle_root(identifier)

    def test_the_managed_root_is_not_the_users_own_models_folder(self, root, monkeypatch):
        """Section 2.2. The models folder is very often another drive shared
        with another front end; eight gigabytes of our download does not go in
        somebody else's directory."""
        monkeypatch.setattr("modules.shared.opts.model_chain_llm_models_dir",
                            str(root / "elsewhere"), raising=False)

        assert mc_llm_paths.models_root() == root / "elsewhere"
        assert mc_llm_paths.managed_models_root() == root / "models" / "managed"

    def test_a_manual_gguf_is_never_moved_or_deleted(self, root, hub, registry, tmp_path):
        mine = tmp_path / "mine.gguf"
        mine.write_bytes(b"my own weights")
        write_state(root, model=str(mine))

        managed.download("test-model")

        assert mine.read_bytes() == b"my own weights"
