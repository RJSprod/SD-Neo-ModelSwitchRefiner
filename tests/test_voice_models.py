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
def _fresh_manifest():
    """The manifest is cached, and several tests here replace it."""
    models.manifest(refresh=True)
    yield
    models._manifest_cache = None
    models._progress.clear()


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

    def test_an_unpinned_bundle_reports_that_rather_than_offering_a_download(self,
                                                                            voice_root):
        """Gate 0 in code. The Whisper and Kokoro artifacts are not pinned in
        this build, so the honest status is "not available" and the honest
        response to a Download press is a refusal, not a hopeful GET."""
        found = models.status()
        assert "not been pinned" in found.stt_message
        with pytest.raises(models.VoiceError, match="pinned"):
            models.install("stt")

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


class TestTheDownloader:
    def test_a_matching_artifact_is_kept(self, tmp_path, monkeypatch):
        payload = b"the bytes this repository describes"
        served = Served(payload)
        monkeypatch.setattr(models.urllib.request, "urlopen", served.open)
        models._download(artifact_for(payload), tmp_path / "thing.onnx", lambda _n: None)
        assert (tmp_path / "thing.onnx").read_bytes() == payload

    def test_a_bad_hash_leaves_nothing_behind(self, tmp_path, monkeypatch):
        payload = b"something else entirely"
        served = Served(payload)
        monkeypatch.setattr(models.urllib.request, "urlopen", served.open)
        artifact = artifact_for(payload, sha256="a" * 64)
        with pytest.raises(models.VoiceError, match="not what this extension"):
            models._download(artifact, tmp_path / "thing.onnx", lambda _n: None)
        assert list(tmp_path.iterdir()) == []

    def test_a_short_file_is_refused(self, tmp_path, monkeypatch):
        payload = b"truncated"
        served = Served(payload)
        monkeypatch.setattr(models.urllib.request, "urlopen", served.open)
        artifact = artifact_for(payload, size=len(payload) + 100)
        with pytest.raises(models.VoiceError, match="bytes"):
            models._download(artifact, tmp_path / "thing.onnx", lambda _n: None)
        assert list(tmp_path.iterdir()) == []

    def test_a_file_longer_than_declared_is_cut_off_rather_than_swallowed(self, tmp_path,
                                                                         monkeypatch):
        """A declared size is a budget as well as a checksum input: a URL that
        started answering with a hundred gigabytes should not fill somebody's
        disk before the hash disagrees."""
        payload = b"x" * 10000
        served = Served(payload)
        monkeypatch.setattr(models.urllib.request, "urlopen", served.open)
        artifact = artifact_for(payload, size=100)
        with pytest.raises(models.VoiceError, match="larger than"):
            models._download(artifact, tmp_path / "thing.onnx", lambda _n: None)
        assert list(tmp_path.iterdir()) == []

    def test_an_unpinned_artifact_is_never_fetched(self, tmp_path, monkeypatch):
        def explode(*args, **kwargs):
            raise AssertionError("an unpinned artifact was requested")

        monkeypatch.setattr(models.urllib.request, "urlopen", explode)
        artifact = artifact_for(b"x", sha256=None)
        with pytest.raises(models.VoiceError, match="not pinned"):
            models._download(artifact, tmp_path / "thing.onnx", lambda _n: None)


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
