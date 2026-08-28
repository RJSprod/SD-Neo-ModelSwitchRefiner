"""The trust root, the download transaction, and the one door to the Internet.

Two claims are being defended here and they are not the same claim.

The first is that nothing arrives that this repository has not already described
byte for byte. Every artifact in the manifest carries an HTTPS URL, a size and a
SHA-256; the downloader checks both and deletes anything that fails; a bundle
becomes "installed" only when a manifest written after verification says so; and
an entry that has not been pinned yet cannot be installed at all rather than
being installed on trust.

The second is that installing the runtime performs no dependency resolution.
That is a different property from "the wheels are hashed", and losing it would
be invisible: pip would quietly reach an index, resolve something nobody
reviewed, and every hash in this repository would still be correct about the two
files it does describe. So there is a test that runs a real ``pip install`` with
a recording HTTP server standing in for the package index, and asserts the
server is never asked for anything.
"""

from __future__ import annotations

import hashlib
import http.server
import json
import shutil
import sys
import threading
import zipfile
from pathlib import Path

import pytest

import mc_voice_models as models
import mc_voice_paths as paths


@pytest.fixture(autouse=True)
def _fresh_manifest(tmp_path, monkeypatch):
    """A manifest of this test's own, cached state cleared around it.

    Copied into ``tmp_path`` rather than read from the repository, because a
    successful install writes its pins into an overlay *beside the manifest* --
    and a test that recorded them beside the real one would leave a file in the
    working tree and pin the next test's fixture behind its back. Which it did.
    """
    private = tmp_path / "voice-manifest"
    private.mkdir()
    copy = private / paths.MANIFEST_FILENAME
    copy.write_text(paths.manifest_path().read_text(encoding="utf-8"), encoding="utf-8")
    monkeypatch.setattr(paths, "manifest_path", lambda: copy)

    models._manifest_cache = None
    models.manifest(refresh=True)
    yield
    models._manifest_cache = None
    models._progress.clear()


def test_the_repository_has_no_pin_overlay_checked_in():
    """The overlay is a machine's own record and is gitignored. One in the tree
    is a test that wrote outside its temporary directory."""
    assert not (paths.extension_root() / paths.MANIFEST_DIRNAME
                / models.LOCAL_PINS_FILENAME).exists()


def rewrite(tmp_path, monkeypatch, change) -> Path:
    """The shipped manifest with one thing changed, on disk, in force."""
    spec = json.loads(paths.manifest_path().read_text(encoding="utf-8"))
    change(spec)
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(spec), encoding="utf-8")
    monkeypatch.setattr(paths, "manifest_path", lambda: path)
    models._manifest_cache = None
    return path


# --------------------------------------------------------------------------- #
# The manifest that ships
# --------------------------------------------------------------------------- #


class TestTheShippedManifest:
    def test_it_parses(self):
        assert models.manifest()["runtime_version"]

    def test_there_is_exactly_one_default_of_each_kind(self):
        spec = models.manifest()
        assert set(spec["defaults"]) == set(paths.KINDS)
        for kind in paths.KINDS:
            assert spec["models"][spec["defaults"][kind]].kind == kind

    def test_every_runtime_artifact_is_pinned_and_served_over_https(self):
        """The runtime closure is the half a maintainer can pin without leaving
        their desk, and it ships pinned. Anything less would mean the first
        thing a user downloads is the thing nobody checked."""
        for platform in models.manifest()["platforms"]:
            assert platform.artifacts, f"{platform.identifier} has no artifacts"
            for artifact in platform.artifacts:
                assert artifact.url.startswith("https://")
                assert artifact.pinned, f"{artifact.filename} is not pinned"
                assert len(artifact.sha256) == 64

    def test_the_runtime_closure_is_complete_for_each_platform(self):
        """R2-2. "Complete" means pip is never asked to find anything: the
        engine and its compiled core are both named, so ``--no-deps`` is a
        correct instruction rather than a broken installation."""
        for platform in models.manifest()["platforms"]:
            names = {artifact.filename for artifact in platform.artifacts}
            assert any("sherpa_onnx-" in name for name in names), platform.identifier
            assert any("sherpa_onnx_core-" in name for name in names), platform.identifier

    def test_windows_and_linux_are_covered_for_the_pythons_forge_uses(self):
        found = {(p.system, p.python) for p in models.manifest()["platforms"]}
        for system in ("windows", "linux"):
            for python in ("3.10", "3.11", "3.12", "3.13"):
                assert (system, python) in found, f"no runtime for {system} {python}"

    def test_every_model_bundle_records_its_licence(self):
        """Kokoro's bundle carries espeak-ng data, which is GPL. Describing the
        whole thing as "Apache" because the weights are would be wrong, and the
        manifest is where that has to be written down."""
        for entry in models.manifest()["models"].values():
            assert entry.license, f"{entry.identifier} has no licence"
        kokoro = models.default_model("tts")
        assert "GPL" in kokoro.license
        assert "espeak" in kokoro.attribution.casefold()

    def test_the_default_voice_is_recorded_even_though_it_cannot_be_chosen(self):
        """Section 30: preserve the identifier so a V2 selector is additive
        rather than a migration."""
        assert models.default_model("tts").voice


class TestWhatTheManifestRefuses:
    def test_an_unknown_identifier(self):
        with pytest.raises(models.VoiceError, match="not a Voice Chat model"):
            models.model("../../etc/passwd")
        with pytest.raises(models.VoiceError):
            models.model("whatever-somebody-typed")

    def test_a_plain_http_url(self, tmp_path, monkeypatch):
        def change(spec):
            spec["models"]["whisper-small-int8"]["files"][0]["url"] = "http://example.com/x"

        rewrite(tmp_path, monkeypatch, change)
        with pytest.raises(models.VoiceError, match="HTTPS"):
            models.manifest(refresh=True)

    def test_a_malformed_hash(self, tmp_path, monkeypatch):
        def change(spec):
            spec["models"]["whisper-small-int8"]["files"][0]["sha256"] = "nonsense"

        rewrite(tmp_path, monkeypatch, change)
        with pytest.raises(models.VoiceError, match="malformed SHA-256"):
            models.manifest(refresh=True)

    def test_a_runtime_platform_with_an_unpinned_wheel(self, tmp_path, monkeypatch):
        def change(spec):
            spec["runtime"]["platforms"][0]["artifacts"][0]["sha256"] = None

        rewrite(tmp_path, monkeypatch, change)
        with pytest.raises(models.VoiceError, match="unpinned"):
            models.manifest(refresh=True)

    def test_a_default_that_is_not_in_the_catalogue(self, tmp_path, monkeypatch):
        def change(spec):
            spec["defaults"]["stt"] = "a-model-nobody-shipped"

        rewrite(tmp_path, monkeypatch, change)
        with pytest.raises(models.VoiceError, match="not in its own catalogue"):
            models.manifest(refresh=True)

    def test_a_bundle_id_that_would_escape_the_voice_folder(self, tmp_path, monkeypatch):
        def change(spec):
            spec["models"]["../escape"] = spec["models"].pop("whisper-small-int8")
            spec["defaults"]["stt"] = "../escape"

        rewrite(tmp_path, monkeypatch, change)
        with pytest.raises((models.VoiceError, ValueError)):
            models.manifest(refresh=True)

    def test_a_newer_schema_is_refused_rather_than_guessed_at(self, tmp_path, monkeypatch):
        rewrite(tmp_path, monkeypatch, lambda spec: spec.update({"schema": 99}))
        with pytest.raises(models.VoiceError, match="schema"):
            models.manifest(refresh=True)


# --------------------------------------------------------------------------- #
# Readiness
# --------------------------------------------------------------------------- #


class TestReadiness:
    def test_a_fresh_machine_is_not_ready(self, voice_root):
        found = models.status()
        assert not found.ready
        assert not found.runtime_ready

    def test_an_unpinned_bundle_is_still_installable(self, voice_root):
        """The correction to a design that shipped a Download button which
        refused to download.

        Whether this repository managed to pin an artifact is a fact about the
        machine the release was cut on, and it has no business appearing in
        front of somebody who has an Internet connection and pressed a button.
        The bundle installs; what changes is who attests to the bytes, which is
        reported rather than hidden."""
        found = models.status()
        assert found.stt_message == "Not installed"
        assert models.refusal("stt") == "", (
            "a bundle this build did not pin is no longer a refusal")

    def test_files_on_disk_are_not_an_installation(self, voice_root):
        """Section 11: installed means the manifest identity and hashes match,
        never that some files happen to be there."""
        entry = models.default_model("stt")
        root = paths.bundle_root("stt", entry.identifier)
        root.mkdir(parents=True)
        for artifact in entry.artifacts:
            (root / artifact.local_name).write_bytes(b"not really an onnx file")
        assert not models.status().stt_ready

    def test_a_manifest_naming_other_hashes_is_not_an_installation(self, voice_root,
                                                                   tmp_path, monkeypatch):
        """The recorded hashes are compared against the catalogue on every
        status call, so an extension update that moves to a new Whisper export
        turns the old bundle into "download it again" rather than into a model
        the manifest no longer describes."""
        def pin(spec):
            for index, entry in enumerate(spec["models"]["whisper-small-int8"]["files"]):
                entry["sha256"] = f"{index:064x}"
                entry["bytes"] = 10

        rewrite(tmp_path, monkeypatch, pin)
        entry = models.default_model("stt")
        root = paths.bundle_root("stt", entry.identifier)
        root.mkdir(parents=True)
        for artifact in entry.artifacts:
            (root / artifact.local_name).write_bytes(b"x")
        (root / paths.INSTALLED_FILENAME).write_text(json.dumps({
            "identifier": entry.identifier,
            "artifacts": {artifact.local_name: "0" * 64 for artifact in entry.artifacts},
        }), encoding="utf-8")
        assert not models.status().stt_ready
        assert "does not match" in models.status().stt_message

    def test_an_unsupported_platform_says_so_instead_of_failing(self, voice_root,
                                                               monkeypatch):
        monkeypatch.setattr(models, "current_platform",
                            lambda: ("haiku", "vax", "3.11"))
        found = models.status()
        assert not found.platform_supported
        assert "no tested CPU runtime" in found.runtime_message

    def test_both_models_are_required_for_ready(self):
        """I-10. Dictation technically needs only Whisper; the microphone is
        still not offered until Kokoro is there, which is the requested
        behaviour and stops a half-installed feature looking finished."""
        make = lambda **kw: models.Status(runtime_message="", stt_message="",  # noqa: E731
                                          tts_message="", platform_supported=True, **kw)
        assert make(runtime_ready=True, stt_ready=True, tts_ready=True).ready
        assert not make(runtime_ready=True, stt_ready=True, tts_ready=False).ready
        assert not make(runtime_ready=True, stt_ready=False, tts_ready=True).ready
        assert not make(runtime_ready=False, stt_ready=True, tts_ready=True).ready


# --------------------------------------------------------------------------- #
# Downloading
# --------------------------------------------------------------------------- #


class Served:
    """A tiny HTTPS-less origin, used only through a patched opener.

    The real :func:`models._download` runs against it, so the size check, the
    running hash and the ``.part`` rename are all the production code rather
    than a re-implementation.
    """

    def __init__(self, payload: bytes):
        self.payload = payload
        self.asked = []

    def open(self, request, timeout=None):
        self.asked.append(getattr(request, "full_url", request))
        import io

        return io.BytesIO(self.payload)


def artifact_for(payload: bytes, **overrides) -> models.Artifact:
    fields = {
        "filename": "thing.onnx",
        "local_name": "thing.onnx",
        "url": "https://example.invalid/thing.onnx",
        "size": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    fields.update(overrides)
    return models.Artifact(**fields)


def expect(payload: bytes, sha256=..., size=..., source="this extension's manifest"):
    import hashlib as _hashlib

    return models.Expected(
        size=len(payload) if size is ... else size,
        sha256=_hashlib.sha256(payload).hexdigest() if sha256 is ... else sha256,
        source=source)


class TestTheDownloader:
    @pytest.fixture
    def landing(self, tmp_path):
        """A directory of this test's own, so "nothing was left behind" is a
        statement about the download rather than about the fixtures."""
        where = tmp_path / "landing"
        where.mkdir()
        return where

    def test_a_matching_artifact_is_kept(self, landing, tmp_path, monkeypatch):
        payload = b"the bytes this repository describes"
        served = Served(payload)
        monkeypatch.setattr(models.urllib.request, "urlopen", served.open)
        models._download(artifact_for(payload), landing / "thing.onnx", lambda _n: None,
                         expect(payload))
        assert (landing / "thing.onnx").read_bytes() == payload

    def test_it_reports_the_digest_of_what_arrived(self, landing, tmp_path, monkeypatch):
        """Which is what lets an unpinned bundle be recorded, so the *second*
        install of it is checked against a constant."""
        import hashlib as _hashlib

        payload = b"whatever the publisher served"
        served = Served(payload)
        monkeypatch.setattr(models.urllib.request, "urlopen", served.open)
        found = models._download(artifact_for(payload, sha256=None), tmp_path / "x",
                                 lambda _n: None,
                                 expect(payload, sha256=None, source="a byte count only"))
        assert found == _hashlib.sha256(payload).hexdigest()

    def test_a_bad_hash_leaves_nothing_behind(self, landing, tmp_path, monkeypatch):
        payload = b"something else entirely"
        served = Served(payload)
        monkeypatch.setattr(models.urllib.request, "urlopen", served.open)
        with pytest.raises(models.VoiceError, match="not what"):
            models._download(artifact_for(payload), landing / "thing.onnx",
                             lambda _n: None, expect(payload, sha256="a" * 64))
        assert list(landing.iterdir()) == []

    def test_a_hash_from_the_publisher_is_enforced_just_as_hard(self, landing, tmp_path,
                                                                monkeypatch):
        """The whole basis for allowing an unpinned bundle to install: a digest
        the publisher gave us is still a digest, and still throws the download
        away when it does not match."""
        payload = b"not what the hub said"
        served = Served(payload)
        monkeypatch.setattr(models.urllib.request, "urlopen", served.open)
        with pytest.raises(models.VoiceError, match="the publisher"):
            models._download(artifact_for(payload, sha256=None), landing / "x",
                             lambda _n: None,
                             expect(payload, sha256="b" * 64,
                                    source="the publisher, over HTTPS"))
        assert list(landing.iterdir()) == []

    def test_a_short_file_is_refused(self, landing, tmp_path, monkeypatch):
        payload = b"truncated"
        served = Served(payload)
        monkeypatch.setattr(models.urllib.request, "urlopen", served.open)
        with pytest.raises(models.VoiceError, match="bytes"):
            models._download(artifact_for(payload), landing / "thing.onnx",
                             lambda _n: None,
                             expect(payload, sha256=None, size=len(payload) + 100))
        assert list(landing.iterdir()) == []

    def test_an_empty_answer_is_refused(self, landing, tmp_path, monkeypatch):
        """A 200 with no body is what a CDN outage looks like, and it would
        otherwise be promoted as a zero-byte model."""
        served = Served(b"")
        monkeypatch.setattr(models.urllib.request, "urlopen", served.open)
        with pytest.raises(models.VoiceError, match="empty file"):
            models._download(artifact_for(b""), landing / "x", lambda _n: None,
                             models.Expected(None, None, "nothing"))
        assert list(landing.iterdir()) == []

    def test_a_file_far_longer_than_expected_is_cut_off(self, landing, tmp_path, monkeypatch):
        """A declared size is a budget as well as a checksum input: a URL that
        started answering with a hundred gigabytes should not fill somebody's
        disk before the hash disagrees."""
        payload = b"x" * 10000
        served = Served(payload)
        monkeypatch.setattr(models.urllib.request, "urlopen", served.open)
        with pytest.raises(models.VoiceError, match="far larger"):
            models._download(artifact_for(payload), landing / "thing.onnx",
                             lambda _n: None, expect(payload, sha256=None, size=100))
        assert list(landing.iterdir()) == []


class TestAskingThePublisherFirst:
    """An artifact this repository has not pinned is resolved before it is
    fetched, rather than refused.

    Shipping a Download button that refuses to download is not a security
    posture; it is a broken feature. What replaces the committed hash is the
    publisher's own digest, read from a HEAD over TLS -- weaker, and labelled
    as weaker everywhere it is reported."""

    class Head:
        def __init__(self, headers, status=200):
            self.headers = headers
            self.status = status
            self.asked = []

        def open(self, request, timeout=None):
            self.asked.append(getattr(request, "get_method", lambda: "GET")())
            outer = self

            class Answer:
                status = outer.status
                headers = outer.headers

                def close(self):
                    pass

            return Answer()

    def test_a_committed_hash_is_used_without_asking_anybody(self, monkeypatch):
        def explode(*args, **kwargs):
            raise AssertionError("a pinned artifact was resolved over the network")

        monkeypatch.setattr(models.urllib.request, "urlopen", explode)
        found = models._resolve(artifact_for(b"x" * 32))
        assert found.verified
        assert found.source == "this extension's manifest"

    def test_the_hubs_lfs_etag_is_taken_as_the_digest(self, monkeypatch):
        """Which is the same fact ``tools/pin_managed_models.py`` reads to pin
        the LLM catalogue — read here at install time instead."""
        head = self.Head({"x-linked-size": "117000000",
                          "x-linked-etag": '"' + "c" * 64 + '"'})
        monkeypatch.setattr(models.urllib.request, "urlopen", head.open)
        found = models._resolve(artifact_for(b"x", sha256=None, size=None))
        assert found.sha256 == "c" * 64
        assert found.size == 117000000
        assert "publisher" in found.source
        assert head.asked == ["HEAD"], "resolving should not download the file"

    def test_a_sha256_prefixed_etag_is_understood(self, monkeypatch):
        head = self.Head({"content-length": "10", "etag": "sha256:" + "d" * 64})
        monkeypatch.setattr(models.urllib.request, "urlopen", head.open)
        assert models._resolve(artifact_for(b"x", sha256=None, size=None)).sha256 == "d" * 64

    def test_an_etag_that_is_not_a_digest_is_not_mistaken_for_one(self, monkeypatch):
        """A release asset on another host answers with an S3-style etag. Saying
        "no digest" is better than pretending one was offered."""
        head = self.Head({"content-length": "4096", "etag": '"abc123-7"'})
        monkeypatch.setattr(models.urllib.request, "urlopen", head.open)
        found = models._resolve(artifact_for(b"x", sha256=None, size=None))
        assert found.sha256 is None
        assert found.verified is False
        assert found.size == 4096
        assert "byte count" in found.source

    def test_a_publisher_that_will_not_answer_does_not_stop_the_install(self,
                                                                       monkeypatch):
        def refuse(*args, **kwargs):
            raise OSError("the network is down")

        monkeypatch.setattr(models.urllib.request, "urlopen", refuse)
        found = models._resolve(artifact_for(b"x", sha256=None, size=None))
        assert found.sha256 is None
        assert "did not answer" in found.source

    def test_an_http_error_is_reported_and_not_fatal(self, monkeypatch):
        head = self.Head({}, status=404)
        monkeypatch.setattr(models.urllib.request, "urlopen", head.open)
        assert "404" in models._resolve(artifact_for(b"x", sha256=None, size=None)).source


class TestPromotion:
    def test_a_previous_install_survives_a_failed_promotion(self, tmp_path, monkeypatch):
        """The one thing a download must never cost is the model that already
        worked. The new bundle cannot land, and the old one is put straight back
        where it was."""
        target = tmp_path / "installed"
        target.mkdir()
        (target / "keep.txt").write_text("the model that already worked", encoding="utf-8")
        staging = tmp_path / "staging"
        staging.mkdir()
        (staging / "new.txt").write_text("the replacement", encoding="utf-8")

        original = Path.rename
        refused = []

        def refuse(self, other):
            # Only the promotion itself fails; putting the old bundle back is
            # allowed, which is the ordinary shape of a transient failure.
            if Path(self) == staging:
                refused.append(other)
                raise OSError("busy")
            return original(self, other)

        monkeypatch.setattr(Path, "rename", refuse)
        with pytest.raises(models.VoiceError, match="Nothing about your installation"):
            models._promote(staging, target)
        monkeypatch.undo()

        assert refused, "the promotion was never attempted"
        assert (target / "keep.txt").exists(), "a failed promotion destroyed a good install"
        assert not (target / "new.txt").exists()

    def test_a_failed_recovery_says_where_the_old_install_went(self, tmp_path, monkeypatch):
        """The worse case, and the one that must not be silent. An installation
        whose model has quietly become ``<name>.previous`` reads as "Voice Chat
        broke itself" unless something says where the files are."""
        target = tmp_path / "installed"
        target.mkdir()
        (target / "keep.txt").write_text("the model that already worked", encoding="utf-8")
        staging = tmp_path / "staging"
        staging.mkdir()

        original = Path.rename

        def refuse(self, other):
            if Path(other) == target:
                raise OSError("busy")
            return original(self, other)

        monkeypatch.setattr(Path, "rename", refuse)
        with pytest.raises(models.VoiceError, match="was not damaged"):
            models._promote(staging, target)
        monkeypatch.undo()

        previous = tmp_path / "installed.previous"
        assert (previous / "keep.txt").read_text(encoding="utf-8") == (
            "the model that already worked")

    def test_a_successful_promotion_replaces_the_whole_directory_at_once(self, tmp_path):
        target = tmp_path / "installed"
        target.mkdir()
        (target / "old.txt").write_text("gone", encoding="utf-8")
        staging = tmp_path / "staging"
        staging.mkdir()
        (staging / "new.txt").write_text("here", encoding="utf-8")

        models._promote(staging, target)
        assert (target / "new.txt").exists()
        assert not (target / "old.txt").exists()
        assert not (tmp_path / "installed.previous").exists()


class TestArchives:
    def test_a_member_that_would_escape_the_bundle_is_dropped(self, tmp_path):
        """A hash says the bytes are the ones named. It says nothing about where
        the paths inside want to be written."""
        import io
        import tarfile

        archive = tmp_path / "bundle.tar.bz2"
        with tarfile.open(archive, "w:bz2") as bundle:
            for name in ("kokoro/model.onnx", "../escaped.txt", "/absolute.txt"):
                data = b"x"
                info = tarfile.TarInfo(name)
                info.size = len(data)
                bundle.addfile(info, io.BytesIO(data))

        destination = tmp_path / "out"
        destination.mkdir()
        artifact = artifact_for(b"x", archive="tar.bz2", strip_root="kokoro")
        models._expand(archive, destination, artifact)

        assert (destination / "model.onnx").exists()
        assert not (tmp_path / "escaped.txt").exists()
        assert not Path("/absolute.txt").exists()

    def test_the_declared_root_is_stripped(self, tmp_path):
        import io
        import tarfile

        archive = tmp_path / "bundle.tar.bz2"
        with tarfile.open(archive, "w:bz2") as bundle:
            info = tarfile.TarInfo("kokoro-multi-lang-v1_0/voices.bin")
            info.size = 1
            bundle.addfile(info, io.BytesIO(b"x"))

        destination = tmp_path / "out"
        destination.mkdir()
        models._expand(archive, destination,
                       artifact_for(b"x", archive="tar.bz2",
                                    strip_root="kokoro-multi-lang-v1_0"))
        assert (destination / "voices.bin").exists()


# --------------------------------------------------------------------------- #
# The runtime, installed offline
# --------------------------------------------------------------------------- #


def build_wheel(directory: Path, name: str = "mcvoicetest", version: str = "1.0") -> Path:
    """The smallest thing pip will install, so a real pip run is cheap."""
    path = directory / f"{name}-{version}-py3-none-any.whl"
    dist = f"{name}-{version}.dist-info"
    records = []
    with zipfile.ZipFile(path, "w") as wheel:
        wheel.writestr(f"{name}/__init__.py", "VALUE = 1\n")
        wheel.writestr(f"{dist}/METADATA",
                       f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n")
        wheel.writestr(f"{dist}/WHEEL",
                       "Wheel-Version: 1.0\nGenerator: test\nRoot-Is-Purelib: true\n"
                       "Tag: py3-none-any\n")
        records.append(f"{name}/__init__.py,,")
        records.append(f"{dist}/METADATA,,")
        records.append(f"{dist}/WHEEL,,")
        records.append(f"{dist}/RECORD,,")
        wheel.writestr(f"{dist}/RECORD", "\n".join(records) + "\n")
    return path


class RecordingIndex:
    """A package index that answers nothing and remembers being asked."""

    def __init__(self):
        self.asked = []

        outer = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):  # noqa: N802 - http.server's own spelling
                outer.asked.append(self.path)
                self.send_response(404)
                self.end_headers()

            do_POST = do_GET

            def log_message(self, *args):
                pass

        self.server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    @property
    def url(self) -> str:
        host, port = self.server.server_address[:2]
        return f"http://{host}:{port}/simple/"

    def close(self):
        self.server.shutdown()
        self.server.server_close()


@pytest.mark.skipif(shutil.which(sys.executable) is None, reason="no interpreter to clone")
class TestTheIsolatedRuntime:
    def test_installing_the_runtime_never_asks_a_package_index(self, tmp_path, monkeypatch):
        """R2-2, proved rather than asserted.

        A real venv, a real ``pip install``, and a real HTTP server configured as
        the index pip would use if it were allowed to use one. If a single
        request arrives, the offline guarantee is not one.
        """
        index = RecordingIndex()
        monkeypatch.setenv("PIP_INDEX_URL", index.url)
        monkeypatch.setenv("PIP_EXTRA_INDEX_URL", index.url)
        try:
            wheels = tmp_path / "wheels"
            wheels.mkdir()
            wheel = build_wheel(wheels)
            platform = models.RuntimePlatform(
                identifier="test", system="linux", machines=("x86_64",), python="3.11",
                artifacts=(artifact_for(b"x", filename=wheel.name, local_name=wheel.name),))

            staging = tmp_path / "staging"
            staging.mkdir()
            models._build_environment(staging, wheels, platform)

            interpreter = staging / "env" / "bin" / "python"
            assert interpreter.exists()
            import subprocess

            proof = subprocess.run([str(interpreter), "-c",
                                    "import mcvoicetest; print(mcvoicetest.VALUE)"],
                                   capture_output=True, text=True, timeout=120)
            assert proof.returncode == 0, proof.stderr
            assert proof.stdout.strip() == "1"
        finally:
            index.close()

        assert index.asked == [], (
            f"the runtime installation reached a package index: {index.asked}")

    def test_the_install_command_disables_the_index_and_the_resolver(self, tmp_path,
                                                                     monkeypatch):
        """The flags, checked separately from the behaviour above, because a
        future change that kept the behaviour by accident is a change that will
        stop keeping it."""
        seen = {}

        class Result:
            returncode = 0
            stdout = ""
            stderr = ""

        def record(command, **kwargs):
            seen["command"] = command
            seen["env"] = kwargs.get("env") or {}
            return Result()

        import venv

        monkeypatch.setattr(models.subprocess, "run", record)
        monkeypatch.setattr(venv.EnvBuilder, "create", lambda self, where: None)

        wheels = tmp_path / "wheels"
        wheels.mkdir()
        wheel = build_wheel(wheels)
        platform = models.RuntimePlatform(
            identifier="test", system="linux", machines=("x86_64",), python="3.11",
            artifacts=(artifact_for(b"x", filename=wheel.name, local_name=wheel.name),))
        staging = tmp_path / "staging"
        (staging / "env" / "bin").mkdir(parents=True)
        (staging / "env" / "bin" / "python").write_text("", encoding="utf-8")
        models._build_environment(staging, wheels, platform)

        command = seen["command"]
        assert "--no-index" in command
        assert "--no-deps" in command
        assert str(wheels / wheel.name) in command
        assert seen["env"]["PIP_NO_INDEX"] == "1"
        assert seen["env"]["PIP_INDEX_URL"] == ""
        # No requirement name anywhere: every installable is a path on this disk.
        assert not any(part == "sherpa-onnx" for part in command)


# --------------------------------------------------------------------------- #
# Saying what it is doing
# --------------------------------------------------------------------------- #


class TestItSaysWhatItIsDoing:
    """A several-hundred-megabyte download with one static line in front of it
    is indistinguishable from a download that has hung. It shipped that way:
    ``on_status`` defaulted to a function that discarded its argument, so every
    sentence the installer wrote went nowhere and the row never moved."""

    def test_a_status_line_reaches_the_progress_record_and_the_log(self, caplog):
        import logging

        caplog.set_level(logging.INFO, logger="model_chain")
        say = models._narrator("stt")
        say("Downloading 2 of 3 — decoder.onnx (262 MB)")

        assert models.progress()["stt"]["text"] == "Downloading 2 of 3 — decoder.onnx (262 MB)"
        assert any("decoder.onnx" in record.getMessage() for record in caplog.records)

    def test_a_caller_that_wants_the_line_gets_it_too(self):
        seen = []
        models._narrator("tts", seen.append)("Expanding…")
        assert seen == ["Expanding…"]

    def test_a_callback_that_raises_does_not_stop_the_install(self):
        def explode(_text):
            raise RuntimeError("the row is gone")

        models._narrator("stt", explode)("still fine")
        assert models.progress()["stt"]["text"] == "still fine"

    def test_a_fraction_is_recorded_without_a_log_line_per_tick(self, caplog):
        import logging

        caplog.set_level(logging.INFO, logger="model_chain")
        tick = models._ticker("stt")
        for step in range(20):
            tick(step / 20.0)

        assert models.progress()["stt"]["fraction"] == pytest.approx(0.95)
        assert not caplog.records, "the progress bar was written to the log"

    def test_a_fraction_is_clamped(self):
        tick = models._ticker("tts")
        tick(-3)
        assert models.progress()["tts"]["fraction"] == 0.0
        tick(9)
        assert models.progress()["tts"]["fraction"] == 1.0

    def test_a_failed_install_records_its_reason_and_logs_it(self, caplog):
        import logging

        caplog.set_level(logging.WARNING, logger="model_chain")
        with pytest.raises(models.VoiceError):
            with models._claim("stt"):
                raise models.VoiceError("decoder.onnx failed its hash check.")

        state = models.progress()["stt"]
        assert state["failed"] is True
        assert state["running"] is False
        assert state["text"] == "decoder.onnx failed its hash check."
        assert any("failed its hash check" in record.getMessage()
                   for record in caplog.records)

    def test_a_second_install_of_the_same_kind_is_refused_while_one_runs(self):
        with models._claim("stt"):
            with pytest.raises(models.VoiceError, match="already being installed"):
                with models._claim("stt"):
                    pass


class TestRefusingBeforeStarting:
    """Everything that can be decided before a thread starts is decided in front
    of the caller, so a browser waiting on the answer gets one."""

    def test_an_unpinned_bundle_is_not_refused_at_all(self):
        """The blocker that shipped, as a test. Nothing about this repository's
        own pinning state may stand between a user and the Download button."""
        assert models.refusal("stt") == ""
        assert models.refusal("tts") == ""

    def test_an_unknown_kind_is_refused(self):
        assert models.refusal("../etc") 
        assert models.refusal("speech")

    def test_an_unsupported_platform_is_refused(self, monkeypatch):
        monkeypatch.setattr(models, "current_platform", lambda: ("haiku", "vax", "3.11"))
        assert "no tested CPU runtime" in models.refusal("stt")

    def test_a_running_install_is_refused(self, monkeypatch):
        monkeypatch.setitem(models._progress, "tts", {"running": True})
        assert "already being installed" in models.refusal("tts")

    def test_a_pinned_and_idle_bundle_is_not_refused(self, tmp_path, monkeypatch):
        def pin(spec):
            for index, entry in enumerate(spec["models"]["whisper-small-int8"]["files"]):
                entry["sha256"] = f"{index:064x}"
                entry["bytes"] = 10

        rewrite(tmp_path, monkeypatch, pin)
        assert models.refusal("stt") == ""


# --------------------------------------------------------------------------- #
# Installing from files somebody already has
# --------------------------------------------------------------------------- #


def onnx_file(path: Path, megabytes: int = 2) -> Path:
    """Something that passes for an ONNX model: the protobuf first byte, and
    enough of it to not look like an error page saved under the wrong name."""
    path.write_bytes(b"\x08" + b"\0" * (megabytes * 1024 * 1024))
    return path


class TestInstallingFromAFolder:
    """The escape hatch, and it earns its place beyond one unpinned build: a
    machine with no route to huggingface.co, a proxy that refuses large
    binaries, an air-gapped install, or somebody who already has these files."""

    @pytest.fixture
    def downloaded(self, tmp_path):
        folder = tmp_path / "Downloads"
        folder.mkdir()
        onnx_file(folder / "small-encoder.int8.onnx")
        onnx_file(folder / "small-decoder.int8.onnx")
        (folder / "small-tokens.txt").write_text("a\nb\nc\n", encoding="utf-8")
        return folder

    def test_it_installs_under_the_names_the_worker_uses(self, voice_root, downloaded):
        """Tolerant about the names coming in, strict about the names going out.
        Telling somebody to rename a publisher's file is this extension making
        its own internal spelling their problem."""
        found = models.install_from("stt", downloaded)

        assert found.stt_ready is True
        root = paths.bundle_root("stt", models.default_id("stt"))
        assert (root / "encoder.onnx").is_file()
        assert (root / "decoder.onnx").is_file()
        assert (root / "tokens.txt").is_file()

    def test_it_says_the_files_were_yours(self, voice_root, downloaded):
        """The honest claim. There is no committed hash for these, so the status
        must not read as though the manifest verified them."""
        models.install_from("stt", downloaded)
        assert "from files you supplied" in models.status().stt_message

    def test_it_records_a_hash_so_later_tampering_still_shows(self, voice_root,
                                                              downloaded):
        models.install_from("stt", downloaded)
        root = paths.bundle_root("stt", models.default_id("stt"))
        record = json.loads((root / paths.INSTALLED_FILENAME).read_text(encoding="utf-8"))
        assert record["source"] == "local"
        assert len(record["artifacts"]["encoder.onnx"]) == 64

    def test_a_file_that_is_not_an_onnx_model_is_refused(self, voice_root, downloaded):
        """The mistake somebody actually makes: a saved HTML error page, or the
        LFS pointer that a plain `git clone` leaves behind."""
        (downloaded / "small-encoder.int8.onnx").write_bytes(
            b"version https://git-lfs.github.com/spec/v1\n" + b"x" * (2 * 1024 * 1024))
        with pytest.raises(models.VoiceError, match="not an ONNX model"):
            models.install_from("stt", downloaded)
        assert not models.status().stt_ready

    def test_a_file_far_too_small_to_be_a_model_is_refused(self, voice_root, downloaded):
        (downloaded / "small-decoder.int8.onnx").write_bytes(b"\x08" + b"\0" * 200)
        with pytest.raises(models.VoiceError, match="too small"):
            models.install_from("stt", downloaded)

    def test_a_missing_file_is_named(self, voice_root, downloaded):
        (downloaded / "small-tokens.txt").unlink()
        with pytest.raises(models.VoiceError, match="tokens.txt"):
            models.install_from("stt", downloaded)

    def test_a_folder_that_is_not_there_says_so(self, voice_root, tmp_path):
        with pytest.raises(models.VoiceError, match="nothing at"):
            models.install_from("stt", tmp_path / "nowhere")

    def test_an_empty_path_is_refused(self, voice_root):
        with pytest.raises(models.VoiceError, match="Give the folder"):
            models.install_from("stt", "   ")

    def test_a_file_inside_the_folder_is_taken_as_the_folder(self, voice_root,
                                                             downloaded):
        """What people paste. Refusing it would be pedantry."""
        models.install_from("stt", downloaded / "small-tokens.txt")
        assert models.status().stt_ready is True

    def test_the_names_this_extension_uses_are_accepted_too(self, voice_root, tmp_path):
        folder = tmp_path / "already-renamed"
        folder.mkdir()
        onnx_file(folder / "encoder.onnx")
        onnx_file(folder / "decoder.onnx")
        (folder / "tokens.txt").write_text("a\n", encoding="utf-8")
        assert models.install_from("stt", folder).stt_ready is True

    def test_files_one_level_down_are_found(self, voice_root, tmp_path, downloaded):
        """An archive extracted with its own top-level directory is what a
        double-click produces, not somebody's mistake."""
        outer = tmp_path / "outer"
        outer.mkdir()
        downloaded.rename(outer / "sherpa-onnx-whisper-small")
        assert models.install_from("stt", outer).stt_ready is True

    def test_a_failed_install_leaves_a_working_one_alone(self, voice_root, downloaded,
                                                         tmp_path):
        models.install_from("stt", downloaded)
        assert models.status().stt_ready is True

        broken = tmp_path / "broken"
        broken.mkdir()
        (broken / "encoder.onnx").write_bytes(b"nope")
        with pytest.raises(models.VoiceError):
            models.install_from("stt", broken)
        assert models.status().stt_ready is True, (
            "a bad folder took away an installation that was working")

    def test_it_leaves_no_staging_directory_behind(self, voice_root, downloaded):
        models.install_from("stt", downloaded)
        staging = paths.staging_root()
        assert not staging.exists() or not any(staging.iterdir())

    def test_a_manual_install_is_not_refused_either(self):
        """There is nothing to download, so there is nothing for a committed
        hash to be about."""
        assert models.refusal("stt", manual=True) == ""

    def test_an_unpacked_kokoro_tree_is_accepted(self, voice_root, tmp_path):
        folder = tmp_path / "kokoro-multi-lang-v1_0"
        folder.mkdir()
        onnx_file(folder / "model.onnx")
        (folder / "voices.bin").write_bytes(b"\0" * 4096)
        (folder / "tokens.txt").write_text("a\n", encoding="utf-8")
        (folder / "lexicon-us-en.txt").write_text("a a\n", encoding="utf-8")
        (folder / "espeak-ng-data").mkdir()
        (folder / "espeak-ng-data" / "phontab").write_bytes(b"\0" * 16)

        assert models.install_from("tts", folder.parent).tts_ready is True
        root = paths.bundle_root("tts", models.default_id("tts"))
        assert (root / "espeak-ng-data" / "phontab").is_file()


class TestTheLocalPinOverlay:
    """Pins a maintainer filled in, kept out of the tracked manifest.

    A checked-in file edited in place turns every later pull into a merge
    conflict, and the first thing anybody does with a conflict in a file full of
    hashes is take one side at random."""

    def test_an_overlay_fills_in_a_missing_hash(self, tmp_path, monkeypatch):
        manifest_path = tmp_path / "managed-voice-models.json"
        manifest_path.write_text(paths.manifest_path().read_text(encoding="utf-8"),
                                 encoding="utf-8")
        (tmp_path / models.LOCAL_PINS_FILENAME).write_text(json.dumps({
            "artifacts": {
                "small-encoder.int8.onnx": {"sha256": "a" * 64, "bytes": 117000000},
            }}), encoding="utf-8")
        monkeypatch.setattr(paths, "manifest_path", lambda: manifest_path)

        models.manifest(refresh=True)
        artifact = models.default_model("stt").artifacts[0]
        assert artifact.sha256 == "a" * 64
        assert artifact.size == 117000000

    def test_it_cannot_change_a_hash_this_repository_committed(self, tmp_path,
                                                               monkeypatch):
        """An overlay that could rewrite a committed hash is an overlay that
        defeats the trust root it is extending."""
        manifest_path = tmp_path / "managed-voice-models.json"
        manifest_path.write_text(paths.manifest_path().read_text(encoding="utf-8"),
                                 encoding="utf-8")
        wheel = models.manifest()["platforms"][0].artifacts[0]
        (tmp_path / models.LOCAL_PINS_FILENAME).write_text(json.dumps({
            "artifacts": {wheel.filename: {"sha256": "b" * 64, "bytes": 99}}}),
            encoding="utf-8")
        monkeypatch.setattr(paths, "manifest_path", lambda: manifest_path)

        models.manifest(refresh=True)
        assert models.manifest()["platforms"][0].artifacts[0].sha256 == wheel.sha256

    def test_a_malformed_overlay_is_ignored_rather_than_fatal(self, tmp_path,
                                                              monkeypatch):
        manifest_path = tmp_path / "managed-voice-models.json"
        manifest_path.write_text(paths.manifest_path().read_text(encoding="utf-8"),
                                 encoding="utf-8")
        (tmp_path / models.LOCAL_PINS_FILENAME).write_text("{ not json",
                                                           encoding="utf-8")
        monkeypatch.setattr(paths, "manifest_path", lambda: manifest_path)
        assert models.manifest(refresh=True)["runtime_version"]

    def test_a_hash_of_the_wrong_shape_is_ignored(self, tmp_path, monkeypatch):
        manifest_path = tmp_path / "managed-voice-models.json"
        manifest_path.write_text(paths.manifest_path().read_text(encoding="utf-8"),
                                 encoding="utf-8")
        (tmp_path / models.LOCAL_PINS_FILENAME).write_text(json.dumps({
            "artifacts": {"small-encoder.int8.onnx": {"sha256": "nope", "bytes": 5}}}),
            encoding="utf-8")
        monkeypatch.setattr(paths, "manifest_path", lambda: manifest_path)

        models.manifest(refresh=True)
        assert models.default_model("stt").artifacts[0].sha256 is None


class TestOneClickInstall:
    """Press the button, get the models. No manual step, no precondition about
    what this repository's release machine could reach."""

    @pytest.fixture
    def hub(self, tmp_path, monkeypatch, voice_root):
        """A publisher that answers a HEAD with a digest and a GET with bytes."""
        import hashlib as _hashlib
        import io as _io

        files = {}
        for name, body in (("small-encoder.int8.onnx", b"\x08" + b"E" * 4096),
                           ("small-decoder.int8.onnx", b"\x08" + b"D" * 4096),
                           ("small-tokens.txt", b"a\nb\n")):
            files[name] = body

        asked = []

        def open_url(request, timeout=None):
            url = getattr(request, "full_url", str(request))
            method = request.get_method() if hasattr(request, "get_method") else "GET"
            name = url.rsplit("/", 1)[-1]
            body = files.get(name)
            asked.append((method, name))
            if body is None:
                raise OSError(f"no such file {name}")
            if method == "HEAD":
                headers = {"x-linked-size": str(len(body)),
                           "x-linked-etag": _hashlib.sha256(body).hexdigest()}

                class Answer:
                    status = 200

                    def close(self):
                        pass

                Answer.headers = headers
                return Answer()
            return _io.BytesIO(body)

        monkeypatch.setattr(models.urllib.request, "urlopen", open_url)
        monkeypatch.setattr(models, "install_runtime",
                            lambda on_status=None, on_progress=None: None)
        return types_namespace(files=files, asked=asked)

    def test_pressing_download_installs_an_unpinned_bundle(self, hub, voice_root):
        found = models.install("stt")

        assert found.stt_ready is True
        root = paths.bundle_root("stt", models.default_id("stt"))
        assert (root / "encoder.onnx").is_file()
        assert (root / "tokens.txt").is_file()

    def test_it_asks_the_publisher_before_it_downloads(self, hub, voice_root):
        models.install("stt")
        methods = [method for method, _name in hub.asked]
        assert methods[0] == "HEAD", "the first thing it did was download"
        assert "GET" in methods

    def test_what_arrived_is_recorded_so_the_next_install_is_pinned(self, hub,
                                                                    voice_root):
        """A bundle this repository could not pin is checked against the
        publisher the first time and against a constant every time after."""
        import hashlib as _hashlib

        models.install("stt")

        overlay = json.loads(models.local_pins_path().read_text(encoding="utf-8"))
        recorded = overlay["artifacts"]["small-encoder.int8.onnx"]
        assert recorded["sha256"] == _hashlib.sha256(
            hub.files["small-encoder.int8.onnx"]).hexdigest()
        assert recorded["bytes"] == len(hub.files["small-encoder.int8.onnx"])

        models.manifest(refresh=True)
        assert models.default_model("stt").artifacts[0].pinned is True

    def test_the_installed_record_says_who_vouched_for_each_file(self, hub, voice_root):
        models.install("stt")
        root = paths.bundle_root("stt", models.default_id("stt"))
        record = json.loads((root / paths.INSTALLED_FILENAME).read_text(encoding="utf-8"))
        assert "publisher" in record["verified_by"]["small-encoder.int8.onnx"]

    def test_a_publisher_digest_that_does_not_match_stops_the_install(self, hub,
                                                                      voice_root,
                                                                      monkeypatch):
        real = models._resolve

        def lie(artifact):
            found = real(artifact)
            return models.Expected(found.size, "f" * 64, found.source)

        monkeypatch.setattr(models, "_resolve", lie)
        with pytest.raises(models.VoiceError, match="not what"):
            models.install("stt")
        assert models.status().stt_ready is False

    def test_a_publisher_that_offers_no_digest_still_installs(self, hub, voice_root,
                                                              monkeypatch):
        """A release asset on a host with no content digest. Size is checked,
        the file is opened, and what arrived is recorded."""
        real = models._resolve
        monkeypatch.setattr(models, "_resolve",
                            lambda a: models.Expected(real(a).size, None,
                                                      "the publisher's byte count only"))
        assert models.install("stt").stt_ready is True

    def test_a_failed_download_installs_nothing(self, hub, voice_root, monkeypatch):
        def refuse(*args, **kwargs):
            raise OSError("the network went away")

        monkeypatch.setattr(models.urllib.request, "urlopen", refuse)
        with pytest.raises(models.VoiceError):
            models.install("stt")

        assert models.status().stt_ready is False
        assert not paths.bundle_root("stt", models.default_id("stt")).exists()

    def test_installing_an_installed_bundle_downloads_nothing(self, hub, voice_root):
        """Nothing should be able to spend somebody's connection re-fetching a
        bundle that is already on their disk, whichever route asked."""
        models.install("stt")
        hub.asked.clear()
        assert models.install("stt").stt_ready is True
        assert hub.asked == []


def types_namespace(**values):
    import types as _types

    return _types.SimpleNamespace(**values)


class TestTheEngineOnItsOwn:
    """Installing both models from files you already have downloads nothing, so
    the engine that both of them need is still missing and there is no model
    button left to press. It gets a button."""

    def test_the_engine_is_an_installable_kind(self):
        assert models.refusal("runtime") == ""

    def test_an_unsupported_platform_still_refuses_it(self, monkeypatch):
        monkeypatch.setattr(models, "current_platform", lambda: ("haiku", "vax", "3.11"))
        assert "no tested CPU runtime" in models.refusal("runtime")

    def test_installing_it_provisions_only_the_runtime(self, voice_root, monkeypatch):
        called = []
        monkeypatch.setattr(models, "install_runtime",
                            lambda on_status=None, on_progress=None: called.append(True))
        models.install_engine()
        assert called == [True]

    def test_it_narrates_into_the_progress_record(self, voice_root, monkeypatch, caplog):
        import logging

        caplog.set_level(logging.INFO, logger="model_chain")
        monkeypatch.setattr(models, "install_runtime",
                            lambda on_status=None, on_progress=None:
                            on_status and on_status("Creating the isolated voice runtime…"))
        models.install_engine()

        state = models.progress()["runtime"]
        assert state["running"] is False
        assert state["failed"] is False
        assert state["text"] == "Installed."
        assert any("Creating the isolated voice runtime" in record.getMessage()
                   for record in caplog.records), "the row was never told what it was doing"

    def test_a_failure_leaves_its_reason_where_the_row_draws_it(self, voice_root,
                                                                monkeypatch):
        def explode(on_status=None, on_progress=None):
            raise models.VoiceError("The isolated interpreter will not run.")

        monkeypatch.setattr(models, "install_runtime", explode)
        with pytest.raises(models.VoiceError):
            models.install_engine()
        assert models.progress()["runtime"]["failed"] is True
        assert "will not run" in models.progress()["runtime"]["text"]


class TestTheSummaryLine:
    def test_it_names_every_missing_piece(self):
        found = models.Status(False, False, False, "n", "n", "n", True)
        assert "the voice engine" in found.summary
        assert "the speech-to-text model" in found.summary
        assert "the text-to-speech model" in found.summary

    def test_it_names_only_what_is_missing(self):
        """The reported confusion: both models installed from files, engine not,
        and three "Not installed" lines that do not say which one matters."""
        found = models.Status(False, True, True, "n", "i", "i", True)
        assert found.summary == ("Voice Chat is not ready yet — still to install: "
                                 "the voice engine.")

    def test_it_says_ready_when_it_is(self):
        assert models.Status(True, True, True, "i", "i", "i", True).summary == (
            "Voice Chat is ready.")

    def test_an_unsupported_platform_says_that_instead(self):
        found = models.Status(False, False, False, "no runtime here", "n", "n", False)
        assert found.summary == "no runtime here"


class TestWhenTheStagedRuntimeWillNotRun:
    """The failure that stopped a real installation, and reported nothing.

    The smoke test ran a subprocess, checked its return code and discarded its
    output, so "could not import its speech engine" was the whole of what
    anybody could learn."""

    def test_a_broken_interpreter_is_told_apart_from_a_broken_engine(self, tmp_path,
                                                                     monkeypatch):
        staging = tmp_path / "staging"
        binary = staging / ("env/Scripts/python.exe" if models.os.name == "nt"
                            else "env/bin/python")
        binary.parent.mkdir(parents=True)
        binary.write_text("", encoding="utf-8")

        class Result:
            returncode = 1
            stdout = ""
            stderr = "No module named 'encodings'"

        monkeypatch.setattr(models.subprocess, "run", lambda *a, **k: Result())
        with pytest.raises(models.VoiceError, match="interpreter will not run"):
            models._smoke_test(staging, models.manifest())

    def test_the_subprocess_output_reaches_the_log(self, tmp_path, monkeypatch, caplog):
        import logging

        caplog.set_level(logging.WARNING, logger="model_chain")
        staging = tmp_path / "staging"
        binary = staging / ("env/Scripts/python.exe" if models.os.name == "nt"
                            else "env/bin/python")
        binary.parent.mkdir(parents=True)
        binary.write_text("", encoding="utf-8")

        class Result:
            returncode = 1
            stdout = ""
            stderr = "ImportError: DLL load failed while importing _sherpa_onnx"

        monkeypatch.setattr(models.subprocess, "run", lambda *a, **k: Result())
        with pytest.raises(models.VoiceError):
            models._smoke_test(staging, models.manifest())

        written = " ".join(record.getMessage() for record in caplog.records)
        assert "DLL load failed" in written, (
            "the subprocess said why it failed and nobody wrote it down")

    def test_a_missing_interpreter_names_the_path(self, tmp_path):
        with pytest.raises(models.VoiceError, match="no interpreter at"):
            models._smoke_test(tmp_path / "staging", models.manifest())

    def test_a_long_traceback_is_trimmed_from_the_front(self):
        """The sentence that matters is the last line of a traceback; the first
        twenty frames are the ones nobody needs."""
        text = "\n".join(f"  frame {i}" for i in range(500)) + "\nValueError: the real reason"
        trimmed = models._quote(text)
        assert trimmed.endswith("ValueError: the real reason")
        assert len(trimmed) < len(text)
