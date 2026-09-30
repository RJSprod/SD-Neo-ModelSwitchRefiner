"""VibeVoice's installer, manifest and settings: the trust root and what it claims.

What is asserted here is the shape of the claim rather than the bytes behind
it, because the bytes cannot be fetched from a test. The manifest names every
PyPI wheel of the closure with a size and a digest and exactly one wheel per
package on the one platform; it carries a versioned resolve for the torch
wheel that cannot be pinned from here; and it says, in its own notes, that the
model, its shards and the tokenizer are declared and not hashed. The installer
is exercised against fake files under a throwaway voice root: status moves
through its states as files appear, the shard list is read from the model's
own index, the processor configuration names the local tokenizer by absolute
path, settings are validated and persisted, and what a render needs stands on
the estimate until a render has been measured.

Nothing here imports torch, transformers or vibevoice. The smoke test, the
environment build and the network are doubles, because the questions are
about transactions, refusals and files rather than tensors.
"""

from __future__ import annotations

import dataclasses
import hashlib
import importlib.util
import io
import json
import re
import struct
import sys
import types
import urllib.error
import urllib.request
from pathlib import Path

import pytest

import mc_voice_models as models
import mc_voice_paths as paths
import mc_voice_vibevoice as vibevoice

ROOT = Path(__file__).resolve().parent.parent
TOOL = ROOT / "tools" / "pin_vibevoice_models.py"

MODEL = "vibevoice-7b"
SHARDS = ("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors")
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt")

FORBIDDEN = ("gradio", "aiortc", "av", "ml-collections", "absl-py", "librosa", "numba",
             "llvmlite", "scipy", "soundfile", "torchaudio", "httpx")
"""What the closure must not ship: the demo's dependencies, the lazily imported
audio file readers and writers, and the HTTP client diffusers' pipeline loaders
want. The manifest's ``runtime.excluded`` names each with its reason."""


def _tool():
    """The pin tool, imported by path: it lives in ``tools/`` and is never on
    the extension's import path. Registered in ``sys.modules`` first, because
    its dataclasses resolve their postponed annotations through it."""
    found = sys.modules.get("pin_vibevoice_models")
    if found is not None:
        return found
    spec = importlib.util.spec_from_file_location("pin_vibevoice_models", TOOL)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _normalise(name: str) -> str:
    return re.sub(r"[-_.]+", "-", str(name)).casefold()


def _safetensors() -> bytes:
    header = json.dumps({"__metadata__": {"format": "pt"}}).encode("utf-8")
    return struct.pack("<Q", len(header)) + header + b"\x00" * 16


def _index(shards=SHARDS) -> dict:
    weights = {}
    for number, name in enumerate(shards):
        weights[f"layer.{number}.weight"] = name
        weights[f"layer.{number}.bias"] = name
    return {"metadata": {"total_size": 1}, "weight_map": weights}


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _files(shards=SHARDS, upstream=True) -> dict:
    """What the mirror and the tokenizer repository would serve, by filename."""
    found = {
        "config.json": json.dumps({"model_type": "vibevoice"}).encode("utf-8"),
        "model.safetensors.index.json": json.dumps(_index(shards)).encode("utf-8"),
    }
    if upstream:
        found["preprocessor_config.json"] = json.dumps({
            "processor_class": "VibeVoiceProcessor", "speech_tok_compress_ratio": 3200,
            "db_normalize": True, "audio_processor": {"sampling_rate": 24000},
            "language_model_pretrained_name": "Qwen/Qwen2.5-7B"}).encode("utf-8")
    for name in shards:
        found[name] = _safetensors()
    found["tokenizer.json"] = json.dumps({"version": "1.0", "model": {}}).encode("utf-8")
    found["tokenizer_config.json"] = json.dumps({"tokenizer_class": "Qwen2Tokenizer"}).encode()
    found["vocab.json"] = json.dumps({"a": 1}).encode("utf-8")
    found["merges.txt"] = b"#version: 0.2\n" + b"a b\n" * 8
    return found


@pytest.fixture(autouse=True)
def fresh():
    vibevoice._manifest_cache_clear()
    yield
    vibevoice._manifest_cache_clear()


@pytest.fixture
def windows(voice_root, monkeypatch):
    """The release target: 64-bit Windows, CPython 3.13, an NVIDIA driver present.

    Without it every status test would land in "no runtime for this platform"
    and prove nothing about the states underneath.
    """
    monkeypatch.setattr(models, "current_platform", lambda: ("windows", "amd64", "3.13"))
    monkeypatch.setattr(vibevoice, "_nvidia_present", lambda: True)
    return voice_root


OVERLAY_PATHS = (
    "vibevoice/modular/configuration_vibevoice.py",
    "vibevoice/modular/configuration_vibevoice_streaming.py",
    "vibevoice/modular/modeling_vibevoice_streaming.py",
    "vibevoice/modular/modeling_vibevoice_streaming_inference.py",
    "vibevoice/processor/vibevoice_streaming_processor.py",
)
OVERLAY_PINS = {
    "vibevoice/modular/configuration_vibevoice.py":
        (16434, "bcf59d8749405d5d25436923444436f6719cb9b2b403941a8a5e46f5b94873d0"),
    "vibevoice/modular/configuration_vibevoice_streaming.py":
        (4667, "1b4417ea2ffe5b42a58d1ee6b3c9508bc253930c2d55f8ec1f10916e36b8029e"),
    "vibevoice/modular/modeling_vibevoice_streaming.py":
        (8031, "96d402716fe2cc6284c4ad7f8b2309dd8d06d2133681e266387466116e908afb"),
    "vibevoice/modular/modeling_vibevoice_streaming_inference.py":
        (42627, "cb5360a62dbf3a836cea763348bc6330c4d7a422fbe572f4abf7b3c7a15bce8b"),
    "vibevoice/processor/vibevoice_streaming_processor.py":
        (18620, "41ff57a7da668db9099c395ab7bbee9b2cd7aa0c2f8f0b0c214768c051aa5a53"),
}
"""The five files of Microsoft's repository at the pinned commit, as hashed from a
clone of it and from raw.githubusercontent.com, which served the same bytes."""
COMMIT = "1541f590c7099820f10ea012f48d2399282df69f"
RAW = f"https://raw.githubusercontent.com/microsoft/VibeVoice/{COMMIT}/"
REALTIME = "vibevoice-realtime-0.5b"


def _overlay_bytes(path: str) -> bytes:
    """What a test serves in place of one overlay file. The real bytes are not
    reachable from a test, so the manifest's entries are lent these bytes' digests."""
    return f'"""{path}, as a test serves it."""\nVALUE = {len(path)}\n'.encode("utf-8")


def _lend_overlay(monkeypatch) -> tuple:
    real = vibevoice.overlay_files()
    lent = tuple(dataclasses.replace(item, artifact=dataclasses.replace(
        item.artifact, size=len(_overlay_bytes(item.path)),
        sha256=_sha(_overlay_bytes(item.path)))) for item in real)
    monkeypatch.setattr(vibevoice, "overlay_files", lambda: lent)
    return lent


class _Net:
    """A network that answers only what a test serves: GET through ``urlopen`` (the
    shared downloader's path) and the publisher HEAD, with a 404 for anything else.
    Every request is recorded, so a test can say what was asked for and what was not."""

    def __init__(self):
        self.served = {}
        self.asked = []

    def urlopen(self, request, timeout=None):
        url = getattr(request, "full_url", request)
        self.asked.append(url)
        if url not in self.served:
            raise urllib.error.HTTPError(url, 404, "Not Found", {}, None)
        return io.BytesIO(self.served[url])

    def head(self, url, hops=5, authorized=False):
        self.asked.append(("HEAD", url))
        if url not in self.served:
            return 404, {}
        return 200, {"content-length": str(len(self.served[url]))}


@pytest.fixture
def net(monkeypatch):
    found = _Net()
    monkeypatch.setattr(urllib.request, "urlopen", found.urlopen)
    monkeypatch.setattr(models, "_head", found.head)
    monkeypatch.setattr(models, "_declared_size", lambda url: None)
    return found


@pytest.fixture
def local_pins(tmp_path, monkeypatch):
    """The untracked overlay of recorded digests, moved out of the repository."""
    where = tmp_path / "managed-vibevoice-models.local.json"
    monkeypatch.setattr(paths, "vibevoice_local_pins_path", lambda: where)
    return where


def _install_runtime_files(closure: str = "") -> None:
    interpreter = vibevoice._interpreter_in(paths.vibevoice_runtime_root())
    interpreter.parent.mkdir(parents=True, exist_ok=True)
    interpreter.write_text("#!/bin/sh\n", encoding="utf-8")
    vibevoice._write_json(paths.vibevoice_runtime_manifest(), {
        "schema": 1, "closure": closure or vibevoice.closure_id(),
        "vibevoice_version": "0.0.1", "torch_version": "2.8.0+cu128",
        "transformers_version": "4.51.3"})


def _install_model_files(shards=SHARDS, tokenizer=True, local_config=True) -> Path:
    root = paths.vibevoice_model_root(MODEL)
    root.mkdir(parents=True, exist_ok=True)
    files = _files(shards)
    for name in ("config.json", "model.safetensors.index.json") + tuple(shards):
        (root / name).write_bytes(files[name])
    if tokenizer:
        folder = root / paths.VIBEVOICE_TOKENIZER_DIRNAME
        folder.mkdir(exist_ok=True)
        for name in TOKENIZER_FILES:
            (folder / name).write_bytes(files[name])
    vibevoice._write_json(root / paths.INSTALLED_FILENAME, {
        "schema": 1, "id": MODEL, "repo": "aoi-ot/VibeVoice-Large", "shards": list(shards),
        "bytes": {name: len(files[name]) for name in shards}})
    if local_config:
        vibevoice._write_json(root / paths.VIBEVOICE_LOCAL_CONFIG, {
            "processor_class": "VibeVoiceProcessor",
            "language_model_pretrained_name": str(root / paths.VIBEVOICE_TOKENIZER_DIRNAME)})
    return root


# --------------------------------------------------------------------------- #
# The manifest
# --------------------------------------------------------------------------- #


class TestTheManifest:
    def test_it_is_schema_one_and_not_pinned(self):
        found = vibevoice.manifest(refresh=True)
        assert found["schema"] == 1
        assert found["pinned"] is False
        assert found["runtime"]["package"] == "vibevoice"
        assert found["runtime"]["import_name"] == "vibevoice"
        assert found["runtime"]["version"] == "0.0.1"
        assert found["defaults"]["model"] == MODEL

    def test_every_closure_triple_has_exactly_one_hashed_wheel_on_the_platform(self):
        """Thirty-odd packages written down, and one real wheel for each.

        A triple without a wheel is a package the installer cannot fetch; a
        triple with two is a closure that has stopped saying which bytes run.
        """
        tool = _tool()
        found = vibevoice.manifest(refresh=True)
        platforms = found["runtime"]["platforms"]
        assert [entry["id"] for entry in platforms] == ["windows-x86_64-cp313-cu128"]
        entry = platforms[0]
        assert entry["system"] == "windows" and entry["python"] == "3.13"
        assert entry["accelerator"] == "cuda"
        wheels = [item for item in entry["artifacts"] if not item.get("resolve")]
        closure = found["runtime"]["closure"]
        assert len(closure) >= 25
        assert len(wheels) == len(closure)
        for name, version, kind in closure:
            matching = [item for item in wheels
                        if _normalise(item["filename"].split("-")[0]) == _normalise(name)
                        and item["filename"].split("-")[1] == version]
            assert len(matching) == 1, (name, version, matching)
            item = matching[0]
            assert tool.classify(item["filename"]) == kind, item["filename"]
            assert re.fullmatch(r"[0-9a-f]{64}", item["sha256"]), item["filename"]
            assert item["bytes"] > 0
            assert item["url"].startswith("https://files.pythonhosted.org/")
            assert item["local_name"] == item["filename"]

    def test_the_platform_carries_the_versioned_torch_resolve(self):
        """The one wheel that cannot be pinned from here, and how it is bounded."""
        found = vibevoice.manifest(refresh=True)
        entry = found["runtime"]["platforms"][0]
        resolves = [item for item in entry["artifacts"] if item.get("resolve")]
        assert len(resolves) == 1
        item = resolves[0]
        assert item["local_name"] == "torch"
        assert item["resolve"]["index"] == "https://download.pytorch.org/whl/cu128/torch/"
        assert item["resolve"]["package"] == "torch"
        assert item["resolve"]["version"] == "2.8.0"
        assert not item.get("url") and not item.get("sha256")
        assert found["runtime"]["torch"]["version"] == "2.8.0"
        assert "torchaudio" not in json.dumps(entry)

    def test_nothing_forbidden_is_in_the_closure_and_each_absence_has_a_reason(self):
        found = vibevoice.manifest(refresh=True)
        names = {_normalise(name) for name, _version, _kind in found["runtime"]["closure"]}
        filenames = " ".join(item.get("filename", "")
                             for item in found["runtime"]["platforms"][0]["artifacts"])
        for name in FORBIDDEN:
            assert _normalise(name) not in names, name
            assert not re.search(rf"(^|\s){re.escape(name.replace('-', '_'))}-\d", filenames)
            assert name in found["runtime"]["excluded"], f"{name} has no recorded reason"
            assert len(found["runtime"]["excluded"][name]) > 20

    def test_the_roots_are_where_the_code_needs_them(self):
        """The versions the community code breaks without, and the bounds
        transformers 4.51.3 puts on everything beside it."""
        found = vibevoice.manifest(refresh=True)
        versions = {_normalise(name): version
                    for name, version, _kind in found["runtime"]["closure"]}
        assert versions["vibevoice"] == "0.0.1"
        assert versions["transformers"] == "4.51.3"
        assert versions["accelerate"] == "1.6.0"
        assert versions["diffusers"].startswith("0.39.")
        assert versions["huggingface-hub"].startswith("0.36.")
        assert versions["tokenizers"].startswith("0.21.")
        assert versions["numpy"].startswith("2.")
        # sympy 1.14 holds mpmath below 1.4; torch 2.8 still asks for
        # setuptools on Python 3.12 and later; tqdm asks for colorama on Windows.
        assert versions["mpmath"].startswith("1.3.")
        assert "setuptools" in versions
        assert "colorama" in versions
        for name in ("safetensors", "regex", "pyyaml", "requests", "filelock", "fsspec",
                     "packaging", "typing-extensions", "sympy", "networkx", "jinja2",
                     "markupsafe", "psutil", "pillow", "tqdm", "certifi", "idna", "urllib3",
                     "charset-normalizer"):
            assert name in versions, name

    def test_the_model_is_declared_and_not_hashed_and_the_notes_say_so(self):
        found = vibevoice.manifest(refresh=True)
        entry = found["models"][MODEL]
        assert entry["repo"] == "aoi-ot/VibeVoice-Large"
        assert entry["mirrors"] == ["vibevoice/VibeVoice-7B"]
        assert entry["license"] == "MIT"
        names = [item["filename"] for item in entry["files"]]
        assert names == ["config.json", "preprocessor_config.json",
                         "model.safetensors.index.json"]
        for item in entry["files"]:
            assert item["sha256"] is None and item["bytes"] is None
            assert item["url"].startswith("https://huggingface.co/aoi-ot/VibeVoice-Large/")
        assert entry["shards"] == []
        tokenizer = entry["tokenizer"]
        assert tokenizer["repo"] == "Qwen/Qwen2.5-7B"
        assert [item["filename"] for item in tokenizer["files"]] == list(TOKENIZER_FILES)
        assert all(item["sha256"] is None for item in tokenizer["files"])
        notes = found["notes"]
        assert "DECLARED BUT NOT HASHED" in notes
        assert "torch" in notes and "download.pytorch.org" in notes
        assert "tokenizer" in notes and "--model" in notes
        assert "librosa" in notes and "httpx" in notes
        assert vibevoice.bundle().estimates["weights_bytes"] == 18_700_000_000
        assert vibevoice._provisional(entry) is True

    def test_the_tokenizer_directory_name_contains_qwen_and_is_the_paths_constant(self):
        """Load-bearing: the processor picks its tokenizer class by that substring."""
        found = vibevoice.manifest(refresh=True)
        dirname = found["models"][MODEL]["tokenizer"]["dirname"]
        assert dirname == paths.VIBEVOICE_TOKENIZER_DIRNAME
        assert "qwen" in dirname.casefold()
        assert "qwen" in vibevoice.bundle().tokenizer.dirname.casefold()
        assert paths.vibevoice_tokenizer_root(MODEL).name == dirname

    def test_the_optional_processor_config_lands_under_the_upstream_name(self):
        entry = vibevoice.bundle()
        upstream = [item for item in entry.artifacts if item.filename == "preprocessor_config.json"]
        assert len(upstream) == 1
        assert upstream[0].local_name == paths.VIBEVOICE_UPSTREAM_CONFIG
        assert entry.optional == (paths.VIBEVOICE_UPSTREAM_CONFIG,)
        assert entry.required_paths == ("config.json", "model.safetensors.index.json")

    def test_a_manifest_that_moves_the_tokenizer_out_of_a_qwen_directory_is_refused(
            self, monkeypatch):
        found = json.loads(json.dumps(vibevoice.manifest(refresh=True)))
        found["models"][MODEL]["tokenizer"]["dirname"] = "tokenizer"
        monkeypatch.setattr(vibevoice, "manifest", lambda refresh=False: found)
        with pytest.raises(vibevoice.VibeVoiceError, match="qwen"):
            vibevoice.bundle()


class TestPinnedAndTheClosureId:
    def test_the_closure_is_pinned_on_the_release_target_and_not_elsewhere(self, windows):
        assert vibevoice.pinned() is True
        chosen = vibevoice.platform()
        assert chosen is not None and len(chosen.artifacts) >= 25
        assert all(item.pinned for item in chosen.artifacts)
        assert not any(item.filename.startswith("torch") for item in chosen.artifacts)

    def test_an_unsupported_machine_has_no_platform(self, voice_root, monkeypatch):
        monkeypatch.setattr(models, "current_platform", lambda: ("linux", "x86_64", "3.13"))
        assert vibevoice.platform() is None
        assert vibevoice.pinned() is False
        assert vibevoice.closure_id() == ""

    def test_the_torch_version_is_part_of_the_closure_id_and_its_digest_is_not(
            self, windows, monkeypatch):
        """Bumping torch makes every installed runtime stale; recording its
        digest later must not."""
        before = vibevoice.closure_id()
        resolve = vibevoice.torch_resolve()
        monkeypatch.setattr(vibevoice, "torch_resolve",
                            lambda: {**resolve, "version": "2.9.1"})
        assert vibevoice.closure_id() != before
        monkeypatch.setattr(vibevoice, "torch_resolve",
                            lambda: {**resolve, "sha256": "c" * 64, "bytes": 5})
        assert vibevoice.closure_id() == before

    def test_full_torch_version_carries_the_index_local_tag(self):
        resolve = {"index": "https://download.pytorch.org/whl/cu128/torch/",
                   "package": "torch", "version": "2.8.0"}
        assert vibevoice.full_torch_version(resolve) == "2.8.0+cu128"
        assert vibevoice.full_torch_version({**resolve, "version": "2.8.0+cu128"}) \
            == "2.8.0+cu128"
        assert _tool().full_torch_version(resolve) == vibevoice.full_torch_version(resolve)


# --------------------------------------------------------------------------- #
# Status
# --------------------------------------------------------------------------- #


class TestStatus:
    def test_an_unsupported_machine_is_told_what_it_needs(self, voice_root, monkeypatch):
        monkeypatch.setattr(models, "current_platform", lambda: ("linux", "x86_64", "3.11"))
        found = vibevoice.status()
        assert found.supported is False and found.ready is False
        assert "needs Windows with an NVIDIA card" in found.message
        assert "linux/x86_64" in found.message
        assert "needs Windows with an NVIDIA card" in vibevoice.refusal()

    def test_a_supported_machine_starts_with_nothing_installed(self, windows):
        found = vibevoice.status()
        assert found.supported is True
        assert not found.runtime_installed and not found.model_installed
        assert not found.tokenizer_installed and not found.ready
        assert found.provisional is True
        assert found.runtime_message.startswith("Not installed — about")
        assert "PyTorch" in found.runtime_message
        assert found.model_message.startswith("Not installed — VibeVoice 7B, about")
        assert found.download_bytes == found.runtime_bytes + found.model_bytes
        assert found.runtime_bytes > 3_000_000_000
        assert found.model_bytes > 18_000_000_000
        assert found.message == "Setup required — VibeVoice's runtime and model still to install."

    def test_the_runtime_counts_as_installed_only_with_the_current_closure(self, windows):
        _install_runtime_files(closure="0123456789abcdef")
        found = vibevoice.status()
        assert not found.runtime_installed
        assert "pins a different VibeVoice runtime" in found.runtime_message
        _install_runtime_files()
        found = vibevoice.status()
        assert found.runtime_installed
        assert found.runtime_message.startswith("Installed — VibeVoice 0.0.1, torch 2.8.0+cu128")
        assert found.download_bytes == found.model_bytes
        vibevoice._interpreter_in(paths.vibevoice_runtime_root()).unlink()
        assert not vibevoice.status().runtime_installed

    def test_the_model_counts_as_installed_with_its_shards_tokenizer_and_config(self, windows):
        root = _install_model_files()
        found = vibevoice.status()
        assert found.model_installed and found.tokenizer_installed
        assert found.model_message.startswith("Installed — VibeVoice 7B, 2 shards")
        (root / SHARDS[1]).unlink()
        found = vibevoice.status()
        assert not found.model_installed
        assert SHARDS[1] in found.model_message and "install the model again" in found.model_message

    def test_a_missing_tokenizer_or_local_config_is_named(self, windows):
        _install_runtime_files()
        root = _install_model_files(tokenizer=False)
        found = vibevoice.status()
        assert found.model_installed and not found.tokenizer_installed
        assert found.ready is False, "a model without its tokenizer cannot render"
        assert "tokenizer is missing" in found.model_message
        assert "tokenizer" in found.message
        assert "tokenizer" in vibevoice.refusal()
        for name in TOKENIZER_FILES:
            (root / paths.VIBEVOICE_TOKENIZER_DIRNAME).mkdir(exist_ok=True)
            (root / paths.VIBEVOICE_TOKENIZER_DIRNAME / name).write_bytes(b"{}" * 8)
        (root / paths.VIBEVOICE_LOCAL_CONFIG).unlink()
        found = vibevoice.status()
        assert not found.model_installed
        assert paths.VIBEVOICE_LOCAL_CONFIG in found.model_message

    def test_ready_needs_all_three_and_the_page_gets_the_parts(self, windows):
        _install_runtime_files()
        _install_model_files()
        found = vibevoice.status()
        assert found.ready and found.message == "Installed."
        assert vibevoice.refusal() == ""
        public = vibevoice.public_status()
        assert public["ready"] and public["installed"] and public["supported"]
        assert public["provisional"] is True and public["pinned"] is True
        assert [part["id"] for part in public["parts"]] == ["runtime", "model"]
        assert all(part["installed"] for part in public["parts"])
        assert public["download_bytes"] == 0
        assert public["settings"]["keep_warm"] is True
        # The fixture's shards are a few bytes each and an install's recorded
        # sizes beat the estimate, so the page's figure is theirs plus the
        # working set -- the same answer the turn request gets.
        assert public["need_vram_bytes"] == vibevoice.need_vram_bytes()
        assert public["need_vram_bytes"] > 2 * 1024 ** 3
        assert public["refusal"] == ""
        json.dumps(public)

    def test_provisional_is_the_manifests_claim_not_the_overlays(self, windows):
        """An earlier install's recorded digests make the *next* download
        checked against a constant; they do not make the repository's claim
        stronger than it is."""
        overlay = paths.vibevoice_local_pins_path()
        original = overlay.read_text(encoding="utf-8") if overlay.exists() else None
        try:
            overlay.write_text(json.dumps({"schema": 1, "artifacts": {
                "aoi-ot/VibeVoice-Large/config.json": {"sha256": "a" * 64, "bytes": 12}}}),
                encoding="utf-8")
            entry = vibevoice.bundle()
            assert entry.artifacts[0].sha256 == "a" * 64, "the overlay fills a blank"
            assert vibevoice.status().provisional is True
        finally:
            if original is None:
                overlay.unlink(missing_ok=True)
            else:
                overlay.write_text(original, encoding="utf-8")

    def test_status_never_raises_on_a_broken_manifest(self, windows, monkeypatch):
        def broken(refresh=False):
            raise vibevoice.VibeVoiceError("The VibeVoice manifest is not readable.")

        monkeypatch.setattr(vibevoice, "manifest", broken)
        found = vibevoice.status()
        assert not found.ready
        assert "not readable" in found.message


# --------------------------------------------------------------------------- #
# The shards, from the model's own index
# --------------------------------------------------------------------------- #


class TestShardsFromTheIndex:
    def test_the_index_drives_the_shard_list_deduplicated_in_its_order(self):
        index = {"weight_map": {
            "a": "model-00003-of-00003.safetensors",
            "b": "model-00001-of-00003.safetensors",
            "c": "model-00003-of-00003.safetensors",
            "d": "model-00002-of-00003.safetensors",
            "e": "model-00001-of-00003.safetensors"}}
        assert vibevoice._shards_named(index) == [
            "model-00003-of-00003.safetensors", "model-00001-of-00003.safetensors",
            "model-00002-of-00003.safetensors"]

    @pytest.mark.parametrize("name", ["../model.safetensors", "sub/model.safetensors",
                                      "model.bin", "", "..\\x.safetensors"])
    def test_a_shard_that_is_not_a_safetensors_file_beside_the_index_is_refused(self, name):
        with pytest.raises(vibevoice.VibeVoiceError, match="not a safetensors file"):
            vibevoice._shards_named({"weight_map": {"w": name}})

    def test_an_index_without_a_weight_map_is_refused(self):
        with pytest.raises(vibevoice.VibeVoiceError, match="weight_map"):
            vibevoice._shards_named({"metadata": {}})
        with pytest.raises(vibevoice.VibeVoiceError, match="weight_map"):
            vibevoice._shards_named({"weight_map": {}})

    def test_the_pin_tool_reads_the_same_index_the_same_way(self):
        tool = _tool()
        index = _index(("model-00002-of-00002.safetensors", "model-00001-of-00002.safetensors"))
        assert tool.shards_named(index) == vibevoice._shards_named(index)
        with pytest.raises(tool.PinError):
            tool.shards_named({"weight_map": {"w": "../x.safetensors"}})

    def test_an_installed_index_is_read_for_the_shards_when_the_marker_has_none(self, windows):
        root = _install_model_files()
        marker = json.loads((root / paths.INSTALLED_FILENAME).read_text(encoding="utf-8"))
        marker.pop("shards")
        (root / paths.INSTALLED_FILENAME).write_text(json.dumps(marker), encoding="utf-8")
        assert vibevoice._installed_shards(root) == list(SHARDS)
        assert vibevoice.status().model_installed


# --------------------------------------------------------------------------- #
# The processor configuration
# --------------------------------------------------------------------------- #


class TestTheLocalConfig:
    def _staged(self, tmp_path, upstream=True) -> Path:
        staging = tmp_path / "staging"
        (staging / paths.VIBEVOICE_TOKENIZER_DIRNAME).mkdir(parents=True)
        files = _files(upstream=upstream)
        for name in TOKENIZER_FILES:
            (staging / paths.VIBEVOICE_TOKENIZER_DIRNAME / name).write_bytes(files[name])
        if upstream:
            (staging / paths.VIBEVOICE_UPSTREAM_CONFIG).write_bytes(
                files["preprocessor_config.json"])
        return staging

    def test_upstream_fields_are_kept_and_the_tokenizer_named_by_absolute_path(
            self, windows, tmp_path):
        staging = self._staged(tmp_path)
        entry = vibevoice.bundle()
        vibevoice._write_local_config(entry, root=staging)
        found = json.loads((staging / paths.VIBEVOICE_LOCAL_CONFIG).read_text(encoding="utf-8"))
        assert found["audio_processor"] == {"sampling_rate": 24000}
        assert found["speech_tok_compress_ratio"] == 3200
        assert found["db_normalize"] is True
        wanted = paths.vibevoice_model_root(MODEL) / paths.VIBEVOICE_TOKENIZER_DIRNAME
        assert found["language_model_pretrained_name"] == str(wanted)
        assert Path(found["language_model_pretrained_name"]).is_absolute()
        assert "qwen" in found["language_model_pretrained_name"].casefold()
        assert "Qwen/Qwen2.5-7B" not in json.dumps(found), "the hub name would be fetched"
        # Upstream's document is kept beside it, untouched.
        upstream = json.loads((staging / paths.VIBEVOICE_UPSTREAM_CONFIG).read_text())
        assert upstream["language_model_pretrained_name"] == "Qwen/Qwen2.5-7B"

    def test_without_an_upstream_document_the_defaults_are_written(self, windows, tmp_path):
        staging = self._staged(tmp_path, upstream=False)
        vibevoice._write_local_config(vibevoice.bundle(), root=staging)
        found = json.loads((staging / paths.VIBEVOICE_LOCAL_CONFIG).read_text(encoding="utf-8"))
        assert found["processor_class"] == "VibeVoiceProcessor"
        assert found["speech_tok_compress_ratio"] == 3200
        assert found["db_normalize"] is True
        assert found["language_model_pretrained_name"].endswith(
            paths.VIBEVOICE_TOKENIZER_DIRNAME)

    def test_it_refuses_when_the_tokenizer_is_not_in_place(self, windows, tmp_path):
        staging = tmp_path / "staging"
        staging.mkdir()
        with pytest.raises(vibevoice.VibeVoiceError, match="tokenizer is not in place"):
            vibevoice._write_local_config(vibevoice.bundle(), root=staging)
        assert not (staging / paths.VIBEVOICE_LOCAL_CONFIG).exists()

    def test_the_installed_model_directory_is_the_default_target(self, windows):
        root = _install_model_files(local_config=False)
        vibevoice._write_local_config(vibevoice.bundle())
        found = json.loads((root / paths.VIBEVOICE_LOCAL_CONFIG).read_text(encoding="utf-8"))
        assert found["language_model_pretrained_name"] == str(
            root / paths.VIBEVOICE_TOKENIZER_DIRNAME)


# --------------------------------------------------------------------------- #
# Settings
# --------------------------------------------------------------------------- #


class TestSettings:
    def test_the_defaults(self, voice_root):
        found = vibevoice.settings()
        assert found == {"card_uuid": "", "model_id": MODEL, "steps": 10, "cfg_scale": 1.3,
                         "seed": None, "max_new_tokens": None, "keep_warm": True,
                         "precision": "bf16", "lora_id": "", "lora_scale": 1.0,
                         "chat": {"card_uuid": "", "precision": "bf16", "lora_id": "",
                                  "lora_scale": 1.0}}

    def test_values_are_validated_and_persisted(self, voice_root):
        found = vibevoice.set_settings({"steps": 12, "cfg_scale": 1.5, "seed": 7,
                                        "max_new_tokens": 4096, "keep_warm": False,
                                        "card_uuid": "GPU-1234-abcd"})
        assert found["steps"] == 12 and found["cfg_scale"] == 1.5 and found["seed"] == 7
        assert found["max_new_tokens"] == 4096 and found["keep_warm"] is False
        assert found["card_uuid"] == "GPU-1234-abcd"
        stored = json.loads(paths.vibevoice_settings_path().read_text(encoding="utf-8"))
        assert stored["steps"] == 12 and stored["keep_warm"] is False
        assert vibevoice.settings()["steps"] == 12
        assert not paths.vibevoice_settings_path().with_name("settings.json.new").exists()

    def test_blank_means_random_and_automatic(self, voice_root):
        found = vibevoice.set_settings({"seed": "", "max_new_tokens": ""})
        assert found["seed"] is None and found["max_new_tokens"] is None
        found = vibevoice.set_settings({"seed": "random", "max_new_tokens": "auto"})
        assert found["seed"] is None and found["max_new_tokens"] is None

    @pytest.mark.parametrize("values, words", [
        ({"steps": 0}, "1 to 50"),
        ({"steps": 51}, "1 to 50"),
        ({"steps": 10.5}, "whole number"),
        ({"cfg_scale": 0.99}, "1.0 to 3.0"),
        ({"cfg_scale": 3.01}, "1.0 to 3.0"),
        ({"cfg_scale": "warm"}, "1.0 to 3.0"),
        ({"seed": -1}, "0 to 2147483647"),
        ({"seed": 2 ** 31}, "0 to 2147483647"),
        ({"max_new_tokens": 0}, "1 to 32768"),
        ({"max_new_tokens": 40000}, "1 to 32768"),
        ({"keep_warm": "maybe"}, "on or off"),
        ({"card_uuid": "GPU 1234; rm"}, "not a card identifier"),
        ({"model_id": "vibevoice-1.5b"}, "not a VibeVoice model"),
        ({"bogus": 1}, "not a VibeVoice setting"),
    ])
    def test_out_of_range_and_unknown_values_are_refused(self, voice_root, values, words):
        with pytest.raises(vibevoice.VibeVoiceError, match=re.escape(words)):
            vibevoice.set_settings(values)
        assert vibevoice.settings()["steps"] == 10, "a refused change wrote nothing"

    def test_a_stored_value_that_no_longer_validates_falls_back_alone(self, voice_root):
        vibevoice._write_json(paths.vibevoice_settings_path(),
                              {"steps": 999, "cfg_scale": 2.0, "model_id": "gone"})
        found = vibevoice.settings()
        assert found["steps"] == 10
        assert found["cfg_scale"] == 2.0
        assert found["model_id"] == MODEL

    def test_keep_warm_defaults_on_and_accepts_the_usual_spellings(self, voice_root):
        assert vibevoice.SETTINGS_DEFAULTS["keep_warm"] is True
        assert vibevoice.set_settings({"keep_warm": "false"})["keep_warm"] is False
        assert vibevoice.set_settings({"keep_warm": 1})["keep_warm"] is True


# --------------------------------------------------------------------------- #
# What a render needs
# --------------------------------------------------------------------------- #


class TestWhatARenderNeeds:
    def test_the_estimate_is_the_weights_plus_the_working_set(self, voice_root):
        assert vibevoice.need_vram_bytes() == 18_700_000_000 + 2 * 1024 ** 3
        assert vibevoice.need_ram_bytes() == 3 * 1024 ** 3

    def test_the_calibration_takes_over_when_it_is_larger(self, voice_root):
        estimate = vibevoice.need_vram_bytes()
        vibevoice.note_peak(MODEL, estimate - 1, 1024)
        assert vibevoice.need_vram_bytes() == estimate, "a smaller peak changes nothing"
        vibevoice.note_peak(MODEL, 25_000_000_000, 4_000_000_000)
        assert vibevoice.need_vram_bytes() == 25_000_000_000
        assert vibevoice.need_ram_bytes() == 4_000_000_000
        vibevoice.note_peak(MODEL, 24_000_000_000, 100)
        assert vibevoice.need_vram_bytes() == 25_000_000_000, "the highest is kept"
        found = vibevoice.calibration()[f"{MODEL}@bf16"]
        assert found["renders"] == 3 and found["peak_bytes"] == 25_000_000_000
        assert found["rss_bytes"] == 4_000_000_000 and found["updated"]

    def test_note_peak_writes_atomically_and_never_raises(self, voice_root):
        vibevoice.note_peak(MODEL, 1_000)
        path = paths.vibevoice_calibration_path()
        assert path.is_file()
        assert not path.with_name(path.name + ".new").exists()
        vibevoice.note_peak("", 5)
        vibevoice.note_peak("../escape", 5)
        vibevoice.note_peak(MODEL, "not a number")
        vibevoice.note_peak(MODEL, 0, 0)
        assert list(vibevoice.calibration()) == [f"{MODEL}@bf16"]
        assert vibevoice.calibration()[f"{MODEL}@bf16"]["renders"] == 1

    def test_committed_shard_sizes_beat_the_estimate(self, voice_root, monkeypatch):
        found = json.loads(json.dumps(vibevoice.manifest(refresh=True)))
        found["models"][MODEL]["shards"] = [
            {"filename": name, "local_name": name, "sha256": "a" * 64, "bytes": 7_000_000_000,
             "url": f"https://huggingface.co/aoi-ot/VibeVoice-Large/resolve/main/{name}"}
            for name in SHARDS]
        monkeypatch.setattr(vibevoice, "manifest", lambda refresh=False: found)
        entry = vibevoice.bundle()
        assert entry.weights_bytes == 14_000_000_000
        assert vibevoice.need_vram_bytes() == 14_000_000_000 + 2 * 1024 ** 3
        assert vibevoice._provisional(found["models"][MODEL]) is True, "the tokenizer is not"

    def test_sizes_an_install_recorded_beat_the_estimate(self, windows):
        _install_model_files()
        entry = vibevoice.bundle()
        recorded = sum(len(_safetensors()) for _ in SHARDS)
        assert vibevoice._weights_bytes(entry) == recorded
        assert vibevoice.need_vram_bytes() == recorded + 2 * 1024 ** 3


# --------------------------------------------------------------------------- #
# Refusals and the worker's environment
# --------------------------------------------------------------------------- #


class TestRefusals:
    def test_the_sentence_names_what_is_missing(self, windows):
        assert vibevoice.refusal() == "VibeVoice's runtime is not installed — install it below."
        _install_runtime_files()
        assert vibevoice.refusal() == "The VibeVoice 7B model is not installed — install it below."
        root = _install_model_files(tokenizer=False)
        assert vibevoice.refusal() == ("The VibeVoice 7B model's tokenizer is not installed — "
                                       "install the model again.")
        (root / paths.VIBEVOICE_TOKENIZER_DIRNAME).mkdir(exist_ok=True)
        for name in TOKENIZER_FILES:
            (root / paths.VIBEVOICE_TOKENIZER_DIRNAME / name).write_bytes(b"{}" * 8)
        assert vibevoice.refusal() == ""

    def test_a_machine_without_an_nvidia_driver_is_refused(self, windows, monkeypatch):
        _install_runtime_files()
        _install_model_files()
        monkeypatch.setattr(vibevoice, "_nvidia_present", lambda: False)
        assert "needs an NVIDIA card" in vibevoice.refusal()

    def test_an_install_in_progress_is_a_refusal(self, windows):
        with models._lock:
            models._progress[vibevoice.KIND] = {"running": True, "text": "…", "fraction": 0.1}
        try:
            assert vibevoice.refusal() == "VibeVoice is still being installed."
        finally:
            with models._lock:
                models._progress.pop(vibevoice.KIND, None)

    def test_an_unpinned_closure_refuses_unless_the_caller_has_a_folder(self, windows,
                                                                        monkeypatch):
        monkeypatch.setattr(vibevoice, "pinned", lambda: False)
        assert "tools/pin_vibevoice_models.py" in vibevoice.refusal()
        assert vibevoice.refusal(manual=True).startswith("VibeVoice's runtime is not installed")


class TestTheWorkerEnvironment:
    def test_it_names_the_card_and_runs_offline(self):
        found = vibevoice.worker_environment("GPU-3f2a-1b2c-3d4e")
        assert found["CUDA_VISIBLE_DEVICES"] == "GPU-3f2a-1b2c-3d4e"
        assert found["HF_HUB_OFFLINE"] == "1" and found["TRANSFORMERS_OFFLINE"] == "1"
        assert found["HF_HUB_DISABLE_TELEMETRY"] == "1"
        assert found["PYTHONNOUSERSITE"] == "1"
        assert found["PYTORCH_CUDA_ALLOC_CONF"] == "expandable_segments:True"
        assert found["OMP_NUM_THREADS"] == "4"
        assert "PYTORCH_NO_CUDA_MEMORY_CACHING" not in found, "it would disable the allocator"
        for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN"):
            assert name not in found

    def test_bare_digits_are_a_card_too_and_an_empty_card_hides_nothing(self):
        assert vibevoice.worker_environment("0")["CUDA_VISIBLE_DEVICES"] == "0"
        assert "CUDA_VISIBLE_DEVICES" not in vibevoice.worker_environment("")

    def test_a_card_identifier_with_a_shell_in_it_is_refused(self):
        with pytest.raises(vibevoice.VibeVoiceError):
            vibevoice.worker_environment("GPU-1; rm -rf /")

    def test_the_worker_script_and_model_directory_are_this_extensions(self, voice_root):
        assert vibevoice.worker_script() == paths.extension_root() / "vibevoice_worker" / "worker.py"
        assert vibevoice.model_dir() == paths.vibevoice_model_root(MODEL)
        with pytest.raises(vibevoice.VibeVoiceError):
            vibevoice.model_dir("../elsewhere")


# --------------------------------------------------------------------------- #
# Torch, from the publisher's index
# --------------------------------------------------------------------------- #


PAGE = """<!DOCTYPE html><html><body>
<a href="/whl/cu128/torch-2.8.0%2Bcu128-cp312-cp312-win_amd64.whl#sha256={cp312}">x</a>
<a href="/whl/cu128/torch-2.8.0%2Bcu128-cp313-cp313-win_amd64.whl#sha256={wanted}">x</a>
<a href="/whl/cu128/torch-2.8.0%2Bcu128-cp313-cp313-manylinux_2_28_x86_64.whl#sha256={linux}">x</a>
<a href="/whl/cu128/torch-2.9.1%2Bcu128-cp313-cp313-win_amd64.whl#sha256={newer}">x</a>
<a href="/whl/cu126/torch-2.8.0%2Bcu126-cp313-cp313-win_amd64.whl#sha256={cu126}">x</a>
</body></html>""".format(cp312="1" * 64, wanted="2" * 64, linux="3" * 64, newer="4" * 64,
                         cu126="5" * 64)


class TestTorchFromTheIndex:
    def test_the_pinned_version_is_chosen_from_a_page_with_decoys(self, windows, monkeypatch):
        asked = []
        monkeypatch.setattr(models, "read_index_page", lambda url: asked.append(url) or PAGE)
        monkeypatch.setattr(models, "_declared_size", lambda url: 3_300_000_001)
        found = vibevoice._resolve_torch(vibevoice.torch_resolve(), vibevoice.platform(),
                                         lambda text: None)
        assert asked == ["https://download.pytorch.org/whl/cu128/torch/"]
        assert found.filename == "torch-2.8.0+cu128-cp313-cp313-win_amd64.whl"
        assert found.url == ("https://download.pytorch.org/whl/cu128/"
                             "torch-2.8.0%2Bcu128-cp313-cp313-win_amd64.whl")
        assert found.sha256 == "2" * 64
        assert found.size == 3_300_000_001
        assert found.pinned

    def test_a_recorded_digest_that_disagrees_with_the_index_is_refused(self, windows,
                                                                         monkeypatch):
        monkeypatch.setattr(models, "read_index_page", lambda url: PAGE)
        resolve = {**vibevoice.torch_resolve(), "sha256": "9" * 64, "bytes": 5}
        with pytest.raises(vibevoice.VibeVoiceError, match="different"):
            vibevoice._resolve_torch(resolve, vibevoice.platform(), lambda text: None)
        agreed = {**vibevoice.torch_resolve(), "sha256": "2" * 64, "bytes": 5}
        found = vibevoice._resolve_torch(agreed, vibevoice.platform(), lambda text: None)
        assert found.size == 5, "a recorded size is used without a HEAD"

    def test_a_page_without_the_version_is_refused(self, windows, monkeypatch):
        monkeypatch.setattr(models, "read_index_page",
                            lambda url: PAGE.replace("2.8.0%2Bcu128-cp313-cp313-win", "x"))
        with pytest.raises(vibevoice.VibeVoiceError):
            vibevoice._resolve_torch(vibevoice.torch_resolve(), vibevoice.platform(),
                                     lambda text: None)

    def test_the_manual_sources_name_the_torch_wheel_and_its_index(self, windows):
        found = vibevoice.sources("runtime")
        torch_rows = [row for row in found if row["filename"].startswith("torch-")]
        assert [row["filename"] for row in torch_rows] == [
            "torch-2.8.0+cu128-cp313-cp313-win_amd64.whl"]
        assert torch_rows[0]["url"] == "https://download.pytorch.org/whl/cu128/torch/"
        assert len(found) == (len(vibevoice.platform().artifacts) + 1
                              + len(vibevoice.overlay_files()))


# --------------------------------------------------------------------------- #
# Installing the model
# --------------------------------------------------------------------------- #


def _fake_fetch(served: dict, calls: list, refuse=()):
    """A stand-in for the phase downloader: writes the served bytes and records
    what was asked for, in order, the way the real one records it."""

    def fetch(entry, artifacts, destination, say, tick, base, share, record, prefix=""):
        artifacts = list(artifacts)
        if not artifacts:
            return
        calls.append([item.filename for item in artifacts])
        destination.mkdir(parents=True, exist_ok=True)
        for item in artifacts:
            if item.filename in refuse or item.filename not in served:
                raise models.VoiceError(f"{item.filename} could not be downloaded (HTTP 404).")
            data = served[item.filename]
            (destination / item.local_name).write_bytes(data)
            name = prefix + item.local_name
            record["digests"][name] = _sha(data)
            record["sizes"][name] = len(data)
            record["verified"][name] = "the publisher, over HTTPS"
            record["repos"][name] = vibevoice._repo_of(item.url)

    return fetch


class TestInstallingTheModel:
    def test_the_download_is_index_driven_and_lands_complete(self, windows, monkeypatch):
        served = _files()
        calls = []
        monkeypatch.setattr(vibevoice, "_fetch", _fake_fetch(served, calls))
        overlay = paths.vibevoice_local_pins_path()
        original = overlay.read_text(encoding="utf-8") if overlay.exists() else None
        try:
            if original is not None:
                overlay.unlink()
            said = []
            vibevoice.install_model(on_status=said.append)
            assert calls == [["config.json", "model.safetensors.index.json"],
                             ["preprocessor_config.json"],
                             list(SHARDS), list(TOKENIZER_FILES)], calls
            root = paths.vibevoice_model_root(MODEL)
            for name in ("config.json", "model.safetensors.index.json") + SHARDS:
                assert (root / name).read_bytes() == served[name]
            for name in TOKENIZER_FILES:
                assert (root / paths.VIBEVOICE_TOKENIZER_DIRNAME / name).read_bytes() == served[name]
            assert (root / paths.VIBEVOICE_UPSTREAM_CONFIG).read_bytes() == served[
                "preprocessor_config.json"]
            local = json.loads((root / paths.VIBEVOICE_LOCAL_CONFIG).read_text(encoding="utf-8"))
            assert local["language_model_pretrained_name"] == str(
                root / paths.VIBEVOICE_TOKENIZER_DIRNAME)
            assert local["audio_processor"] == {"sampling_rate": 24000}
            marker = json.loads((root / paths.INSTALLED_FILENAME).read_text(encoding="utf-8"))
            assert marker["shards"] == list(SHARDS)
            assert marker["repo"] == "aoi-ot/VibeVoice-Large"
            assert marker["source"] == "hub"
            assert marker["digests"]["config.json"] == _sha(served["config.json"])
            assert marker["digests"][f"{paths.VIBEVOICE_TOKENIZER_DIRNAME}/merges.txt"] \
                == _sha(served["merges.txt"])
            assert marker["bytes"][SHARDS[0]] == len(served[SHARDS[0]])
            assert marker["verified_by"][SHARDS[0]] == "the publisher, over HTTPS"
            assert not paths.vibevoice_staging_root().exists() or \
                not any(paths.vibevoice_staging_root().iterdir())
            found = vibevoice.status()
            assert found.model_installed and found.tokenizer_installed
            # What arrived is recorded for the next install, keyed by repository
            # and filename, and only for what the manifest could not hash.
            pins = json.loads(overlay.read_text(encoding="utf-8"))["artifacts"]
            assert pins["aoi-ot/VibeVoice-Large/config.json"] == {
                "sha256": _sha(served["config.json"]), "bytes": len(served["config.json"])}
            assert pins[f"aoi-ot/VibeVoice-Large/{SHARDS[1]}"]["sha256"] == _sha(served[SHARDS[1]])
            assert pins["Qwen/Qwen2.5-7B/tokenizer.json"]["sha256"] == _sha(served["tokenizer.json"])
            assert vibevoice.bundle().artifacts[0].sha256 == _sha(served["config.json"])
            assert any("names 2 shards" in text for text in said)
        finally:
            if original is None:
                overlay.unlink(missing_ok=True)
            else:
                overlay.write_text(original, encoding="utf-8")

    def test_a_missing_upstream_config_is_skipped_and_the_defaults_written(self, windows,
                                                                             monkeypatch):
        served = _files(upstream=False)
        calls = []
        monkeypatch.setattr(vibevoice, "_fetch", _fake_fetch(served, calls))
        monkeypatch.setattr(vibevoice, "_record_pins", lambda entry, record: None)
        vibevoice.install_model()
        root = paths.vibevoice_model_root(MODEL)
        assert not (root / paths.VIBEVOICE_UPSTREAM_CONFIG).exists()
        local = json.loads((root / paths.VIBEVOICE_LOCAL_CONFIG).read_text(encoding="utf-8"))
        assert local["processor_class"] == "VibeVoiceProcessor"
        assert local["speech_tok_compress_ratio"] == 3200
        assert vibevoice.status().ready is False and vibevoice.status().model_installed

    def test_a_shard_that_will_not_arrive_installs_nothing(self, windows, monkeypatch):
        served = _files()
        monkeypatch.setattr(vibevoice, "_fetch", _fake_fetch(served, [], refuse=(SHARDS[1],)))
        with pytest.raises(models.VoiceError):
            vibevoice.install_model()
        assert not paths.vibevoice_model_root(MODEL).exists()
        assert not vibevoice.status().model_installed

    def test_a_manifest_that_lists_shards_refuses_an_index_naming_others(self, windows,
                                                                          monkeypatch):
        found = json.loads(json.dumps(vibevoice.manifest(refresh=True)))
        found["models"][MODEL]["shards"] = [
            {"filename": "model-00001-of-00001.safetensors",
             "local_name": "model-00001-of-00001.safetensors", "sha256": "a" * 64, "bytes": 5,
             "url": "https://huggingface.co/aoi-ot/VibeVoice-Large/resolve/main/"
                    "model-00001-of-00001.safetensors"}]
        monkeypatch.setattr(vibevoice, "manifest", lambda refresh=False: found)
        entry = vibevoice.bundle()
        with pytest.raises(vibevoice.VibeVoiceError, match="not recorded"):
            vibevoice._shard_artifacts(entry, list(SHARDS))
        listed = vibevoice._shard_artifacts(entry, ["model-00001-of-00001.safetensors"])
        assert listed[0].sha256 == "a" * 64, "a listed shard is checked against its pin"

    def test_install_from_a_folder_with_a_tokenizer_subfolder(self, windows, tmp_path,
                                                              monkeypatch):
        served = _files()
        folder = tmp_path / "downloads"
        (folder / paths.VIBEVOICE_TOKENIZER_DIRNAME).mkdir(parents=True)
        for name in ("config.json", "preprocessor_config.json",
                     "model.safetensors.index.json") + SHARDS:
            (folder / name).write_bytes(served[name])
        for name in TOKENIZER_FILES:
            (folder / paths.VIBEVOICE_TOKENIZER_DIRNAME / name).write_bytes(served[name])

        def never(*args, **kwargs):
            raise AssertionError("a folder install reached for the network")

        monkeypatch.setattr(vibevoice, "_fetch", never)
        found = vibevoice.install_from("model", str(folder))
        assert found.model_installed and found.tokenizer_installed
        root = paths.vibevoice_model_root(MODEL)
        marker = json.loads((root / paths.INSTALLED_FILENAME).read_text(encoding="utf-8"))
        assert marker["source"] == "local" and marker["source_folder"] == str(folder)
        assert marker["shards"] == list(SHARDS)
        assert marker["verified_by"]["config.json"] == "your own files"
        assert (root / paths.VIBEVOICE_UPSTREAM_CONFIG).is_file()
        local = json.loads((root / paths.VIBEVOICE_LOCAL_CONFIG).read_text(encoding="utf-8"))
        assert local["language_model_pretrained_name"] == str(
            root / paths.VIBEVOICE_TOKENIZER_DIRNAME)
        assert not paths.vibevoice_local_pins_path().exists() or "downloads" not in \
            paths.vibevoice_local_pins_path().read_text(encoding="utf-8")

    def test_a_folder_missing_a_shard_the_index_names_is_refused(self, windows, tmp_path):
        served = _files()
        folder = tmp_path / "downloads"
        folder.mkdir()
        for name in ("config.json", "model.safetensors.index.json", SHARDS[0]):
            (folder / name).write_bytes(served[name])
        with pytest.raises(vibevoice.VibeVoiceError, match=re.escape(SHARDS[1])):
            vibevoice.install_from("model", str(folder))
        assert not paths.vibevoice_model_root(MODEL).exists()

    def test_a_folder_without_a_tokenizer_fetches_it_and_says_where_to_put_it(
            self, windows, tmp_path, monkeypatch):
        served = _files()
        folder = tmp_path / "downloads"
        folder.mkdir()
        for name in ("config.json", "model.safetensors.index.json") + SHARDS:
            (folder / name).write_bytes(served[name])
        calls = []
        monkeypatch.setattr(vibevoice, "_fetch", _fake_fetch(served, calls))
        monkeypatch.setattr(vibevoice, "_record_pins", lambda entry, record: None)
        vibevoice.install_from("model", str(folder))
        assert calls == [list(TOKENIZER_FILES)]
        assert vibevoice.status().tokenizer_installed
        vibevoice.uninstall()
        monkeypatch.setattr(vibevoice, "_fetch", _fake_fetch({}, [], refuse=TOKENIZER_FILES))
        with pytest.raises(vibevoice.VibeVoiceError, match=paths.VIBEVOICE_TOKENIZER_DIRNAME):
            vibevoice.install_from("model", str(folder))

    def test_the_repair_path_rewrites_only_the_local_config(self, windows, monkeypatch):
        _install_model_files(local_config=False)

        def never(*args, **kwargs):
            raise AssertionError("a repair downloaded something")

        monkeypatch.setattr(vibevoice, "_fetch", never)
        said = []
        vibevoice.install_model(on_status=said.append)
        assert vibevoice.status().model_installed
        assert any("processor configuration" in text for text in said)
        vibevoice.install_model(on_status=said.append)
        assert said[-1] == "VibeVoice 7B is already installed."

    def test_an_unknown_part_is_refused(self, windows):
        with pytest.raises(vibevoice.VibeVoiceError):
            vibevoice.install("voices")
        with pytest.raises(vibevoice.VibeVoiceError):
            vibevoice.install_from("cloning", "/nowhere")


# --------------------------------------------------------------------------- #
# Installing the runtime
# --------------------------------------------------------------------------- #


class TestInstallingTheRuntime:
    @pytest.fixture
    def doubles(self, windows, monkeypatch, net):
        """Every step that needs the network, a venv or a real interpreter, replaced.

        The overlay is lent digests of bytes the fake network serves, and the
        doubled build leaves a package with the wheel's own copy of the one module
        the overlay replaces, so what is written over what can be seen.
        """
        state = {"fetched": [], "built": None, "smoked": None, "smoked_overlay": None}
        state["overlay"] = _lend_overlay(monkeypatch)
        for item in state["overlay"]:
            net.served[item.artifact.url] = _overlay_bytes(item.path)
        state["net"] = net
        torch_item = models.Artifact(
            filename="torch-2.8.0+cu128-cp313-cp313-win_amd64.whl",
            local_name="torch-2.8.0+cu128-cp313-cp313-win_amd64.whl",
            url="https://download.pytorch.org/whl/cu128/torch-2.8.0%2Bcu128-cp313-cp313-win_amd64.whl",
            size=12, sha256="2" * 64)
        monkeypatch.setattr(vibevoice, "_resolve_torch",
                            lambda resolve, chosen, say: torch_item)
        monkeypatch.setattr(models, "_expectations", lambda artifacts, say: {
            item.local_name: models.Expected(item.size, item.sha256, "this extension's manifest")
            for item in artifacts})
        monkeypatch.setattr(models, "_make_room", lambda *args, **kwargs: None)

        def fetch_all(artifacts, destination, say, tick, budget, expectations):
            destination.mkdir(parents=True, exist_ok=True)
            for item in artifacts:
                (destination / item.local_name).write_bytes(b"wheel " + item.filename.encode())
                state["fetched"].append(item.filename)
            return {item.filename: item.sha256 for item in artifacts}

        monkeypatch.setattr(models, "_fetch_all", fetch_all)

        def build(staging, wheels, chosen):
            state["built"] = [item.filename for item in chosen.artifacts]
            interpreter = vibevoice._interpreter_in(staging)
            interpreter.parent.mkdir(parents=True, exist_ok=True)
            interpreter.write_text("#!/bin/sh\n", encoding="utf-8")
            package = models.site_packages(staging / "env") / "vibevoice" / "modular"
            package.mkdir(parents=True, exist_ok=True)
            (package / "configuration_vibevoice.py").write_bytes(b"# the wheel's own copy\n")

        monkeypatch.setattr(vibevoice, "_build_environment", build)

        def smoke(staging):
            state["smoked"] = staging
            target = models.site_packages(staging / "env")
            state["smoked_overlay"] = {path: (target / path).read_bytes()
                                       for path in OVERLAY_PATHS if (target / path).is_file()}
            return {"ok": True, "torch": "2.8.0+cu128", "transformers": "4.51.3",
                    "vibevoice": "0.0.1", "cuda": True}

        monkeypatch.setattr(vibevoice, "_smoke_test", smoke)
        return state

    def test_the_runtime_is_built_from_the_pinned_wheels_and_the_resolved_torch(self, doubles):
        said = []
        vibevoice.install_runtime(on_status=said.append)
        wheels = [item.filename for item in vibevoice.platform().artifacts]
        assert doubles["fetched"] == wheels + ["torch-2.8.0+cu128-cp313-cp313-win_amd64.whl"]
        assert doubles["built"] == doubles["fetched"]
        assert doubles["smoked"] is not None
        record = json.loads(paths.vibevoice_runtime_manifest().read_text(encoding="utf-8"))
        assert record["closure"] == vibevoice.closure_id()
        assert record["torch_version"] == "2.8.0+cu128"
        assert record["vibevoice_version"] == "0.0.1"
        assert record["transformers_version"] == "4.51.3"
        assert record["torch_wheel"]["filename"] == "torch-2.8.0+cu128-cp313-cp313-win_amd64.whl"
        assert record["torch_wheel"]["sha256"] == _sha(
            b"wheel torch-2.8.0+cu128-cp313-cp313-win_amd64.whl")
        assert record["accelerator"] == "cuda"
        assert set(record["artifacts"]) == set(wheels)
        assert vibevoice.runtime_python() == vibevoice._interpreter_in(
            paths.vibevoice_runtime_root())
        assert not (paths.vibevoice_runtime_root() / "wheels").exists()
        assert vibevoice.status().runtime_installed
        assert said[-1] == "The VibeVoice runtime is installed."

    def test_an_installed_runtime_with_the_current_closure_returns_early(self, doubles):
        vibevoice.install_runtime()
        doubles["fetched"].clear()
        said = []
        vibevoice.install_runtime(on_status=said.append)
        assert said == ["The VibeVoice runtime is already installed."]
        assert doubles["fetched"] == []

    def test_a_failed_smoke_test_promotes_nothing(self, doubles, monkeypatch):
        def refuse(staging):
            raise vibevoice.VibeVoiceError("The staged VibeVoice runtime could not import its "
                                           "packages (ModuleNotFoundError: torch). Nothing was "
                                           "installed.")

        monkeypatch.setattr(vibevoice, "_smoke_test", refuse)
        with pytest.raises(vibevoice.VibeVoiceError, match="could not import"):
            vibevoice.install_runtime()
        assert not paths.vibevoice_runtime_root().exists()
        assert not vibevoice.status().runtime_installed

    def test_install_everything_runs_the_runtime_then_the_model(self, windows, monkeypatch):
        order = []
        monkeypatch.setattr(vibevoice, "install_runtime",
                            lambda on_status=None, on_progress=None, folder=None:
                            order.append("runtime"))
        monkeypatch.setattr(vibevoice, "install_model",
                            lambda on_status=None, on_progress=None, folder=None,
                            model_id="": order.append("model"))
        vibevoice.install()
        assert order == ["runtime", "model"]
        vibevoice.install("model")
        assert order == ["runtime", "model", "model"]
        assert vibevoice.progress()["running"] is False
        assert vibevoice.progress()["text"] == "Installed."

    def test_an_unsupported_platform_cannot_install(self, voice_root, monkeypatch):
        monkeypatch.setattr(models, "current_platform", lambda: ("linux", "x86_64", "3.13"))
        with pytest.raises(vibevoice.VibeVoiceError, match="needs Windows with an NVIDIA card"):
            vibevoice.install_runtime()

    def test_the_smoke_test_reads_the_workers_report_and_hides_paths(self, windows, tmp_path,
                                                                     monkeypatch):
        class Result:
            def __init__(self, code, out, err=""):
                self.returncode, self.stdout, self.stderr = code, out, err

        answers = iter([Result(0, "3.13.0\n"),
                        Result(1, json.dumps({"ok": False, "error":
                                              "ModuleNotFoundError: No module named 'torch' "
                                              "(C:\\Users\\me\\webui\\site-packages\\torch.py)"}))])
        monkeypatch.setattr(vibevoice, "_run_staged", lambda *args, **kwargs: next(answers))
        monkeypatch.setattr(vibevoice, "worker_script", lambda: Path(__file__))
        with pytest.raises(vibevoice.VibeVoiceError) as caught:
            vibevoice._smoke_test(tmp_path)
        assert "No module named 'torch'" in str(caught.value)
        assert "C:\\Users" not in str(caught.value)
        answers = iter([Result(0, "3.13.0\n"),
                        Result(0, "noise\n" + json.dumps({"ok": True, "torch": "2.8.0+cu128",
                                                           "vibevoice": "0.0.1"}) + "\n")])
        monkeypatch.setattr(vibevoice, "_run_staged", lambda *args, **kwargs: next(answers))
        assert vibevoice._smoke_test(tmp_path)["torch"] == "2.8.0+cu128"

    def test_a_missing_worker_script_is_a_refusal_before_anything_runs(self, windows,
                                                                       tmp_path, monkeypatch):
        monkeypatch.setattr(vibevoice, "worker_script", lambda: tmp_path / "absent.py")
        with pytest.raises(vibevoice.VibeVoiceError, match="worker script is missing"):
            vibevoice._smoke_test(tmp_path)

    def test_the_runtime_from_a_folder_needs_the_torch_wheel_too(self, windows, tmp_path,
                                                                 monkeypatch, doubles):
        folder = tmp_path / "wheels"
        folder.mkdir()
        chosen = vibevoice.platform()
        wheel_bytes = {}
        for item in chosen.artifacts:
            data = b"wheel " + item.filename.encode()
            wheel_bytes[item.filename] = data
            (folder / item.filename).write_bytes(data)
        # The pinned wheels are checked against the committed digests, which a
        # fake cannot match -- so this test lends them the fake's digests.
        lent = models.RuntimePlatform(
            identifier=chosen.identifier, system=chosen.system, machines=chosen.machines,
            python=chosen.python,
            artifacts=tuple(models.Artifact(filename=item.filename, local_name=item.local_name,
                                            url=item.url, size=len(wheel_bytes[item.filename]),
                                            sha256=_sha(wheel_bytes[item.filename]))
                            for item in chosen.artifacts))
        monkeypatch.setattr(vibevoice, "platform", lambda: lent)
        with pytest.raises(vibevoice.VibeVoiceError, match="torch-2.8.0\\+cu128"):
            vibevoice.install_runtime(folder=str(folder))
        (folder / "torch-2.8.0+cu128-cp313-cp313-win_amd64.whl").write_bytes(b"torch wheel")
        # The overlay files, saved under their own names beside the wheels.
        for path in OVERLAY_PATHS:
            (folder / path.rsplit("/", 1)[-1]).write_bytes(_overlay_bytes(path))
        vibevoice.install_runtime(folder=str(folder))
        record = json.loads(paths.vibevoice_runtime_manifest().read_text(encoding="utf-8"))
        assert record["torch_wheel"]["sha256"] == _sha(b"torch wheel")
        assert doubles["fetched"] == [], "a folder install downloads nothing"


    def test_the_overlay_is_written_over_the_package_before_the_self_test(self, doubles):
        vibevoice.install_runtime()
        target = models.site_packages(paths.vibevoice_runtime_root() / "env")
        for item in doubles["overlay"]:
            assert (target / item.path).read_bytes() == _overlay_bytes(item.path)
        assert doubles["smoked_overlay"] == {path: _overlay_bytes(path) for path in OVERLAY_PATHS}, \
            "the self-test ran against the code the worker will run"
        record = json.loads(paths.vibevoice_runtime_manifest().read_text(encoding="utf-8"))
        assert record["overlay"] == {item.path: item.artifact.sha256
                                     for item in doubles["overlay"]}
        assert record["overlay_source"] == {"repository": "https://github.com/microsoft/VibeVoice",
                                            "commit": COMMIT}
        assert [url for url in doubles["net"].asked if str(url).startswith(RAW)] == [
            RAW + path for path in OVERLAY_PATHS]
        assert not (paths.vibevoice_runtime_root() / "overlay").exists()
        assert vibevoice.status().runtime_installed

    def test_what_the_self_test_could_do_is_recorded_and_said(self, doubles, monkeypatch):
        """PEFT and bitsandbytes are reported by the self-test, not required by it; the
        record keeps the answer and the runtime's line says what is unavailable."""
        monkeypatch.setattr(vibevoice, "_smoke_test", lambda staging: {
            "ok": True, "torch": "2.8.0+cu128", "transformers": "4.51.3", "vibevoice": "0.0.1",
            "cuda": True, "realtime": True, "lora": True, "peft": "0.17.1",
            "quantisation": False, "bitsandbytes": ""})
        vibevoice.install_runtime()
        record = json.loads(paths.vibevoice_runtime_manifest().read_text(encoding="utf-8"))
        assert record["features"] == {"realtime": True, "lora": True, "quantisation": False}
        assert (record["peft_version"], record["bitsandbytes_version"]) == ("0.17.1", "")
        found = vibevoice.status()
        assert found.runtime_installed
        assert found.runtime_message.endswith("8-bit and 4-bit are unavailable: bitsandbytes "
                                              "did not import when it was checked.")
        assert "LoRA adapters are unavailable" not in found.runtime_message

    def test_an_overlay_file_that_arrives_wrong_installs_nothing(self, doubles):
        url = doubles["overlay"][3].artifact.url
        good = doubles["net"].served[url]
        doubles["net"].served[url] = good[:-2] + b"!\n"
        with pytest.raises(models.VoiceError, match="not what"):
            vibevoice.install_runtime()
        assert not paths.vibevoice_runtime_root().exists()
        assert doubles["smoked"] is None, "nothing unverified ever ran"
        assert doubles["built"] is None, "refused before the runtime was built"

    def test_the_overlay_is_part_of_the_closure_and_a_runtime_without_it_is_stale(
            self, doubles, monkeypatch):
        vibevoice.install_runtime()
        before = vibevoice.closure_id()
        changed = tuple(dataclasses.replace(item, artifact=dataclasses.replace(
            item.artifact, sha256="e" * 64)) if index == 2 else item
                        for index, item in enumerate(doubles["overlay"]))
        monkeypatch.setattr(vibevoice, "overlay_files", lambda: changed)
        assert vibevoice.closure_id() != before
        found = vibevoice.status()
        assert not found.runtime_installed and found.runtime_stale
        assert "installed by an earlier build" in vibevoice.refusal()
        assert "earlier build" in vibevoice.model_info(MODEL)["message"]

    def test_a_package_the_overlay_cannot_complete_is_refused(self, doubles, tmp_path):
        staging = tmp_path / "staging"
        models.site_packages(staging / "env").mkdir(parents=True)
        with pytest.raises(vibevoice.VibeVoiceError, match="no vibevoice package"):
            vibevoice._apply_overlay(staging, tmp_path / "patches", doubles["overlay"],
                                     lambda text: None)
        (models.site_packages(staging / "env") / "vibevoice").mkdir()
        with pytest.raises(vibevoice.VibeVoiceError, match="missing from the staged download"):
            vibevoice._apply_overlay(staging, tmp_path / "patches", doubles["overlay"],
                                     lambda text: None)
        for item in doubles["overlay"]:
            (tmp_path / "patches" / item.path).parent.mkdir(parents=True, exist_ok=True)
            (tmp_path / "patches" / item.path).write_bytes(_overlay_bytes(item.path))
        (tmp_path / "patches" / OVERLAY_PATHS[2]).write_bytes(b"# changed after it was checked\n")
        with pytest.raises(vibevoice.VibeVoiceError, match="did not read back as the pinned"):
            vibevoice._apply_overlay(staging, tmp_path / "patches", doubles["overlay"],
                                     lambda text: None)
        stray = dataclasses.replace(doubles["overlay"][0], path="vibevoice/../sitecustomize.py")
        with pytest.raises(vibevoice.VibeVoiceError, match="outside the vibevoice package"):
            vibevoice._apply_overlay(staging, tmp_path / "patches", (stray,), lambda text: None)
        assert not (models.site_packages(staging / "env") / "sitecustomize.py").exists()

    @pytest.mark.parametrize("layout", ["tree", "clone", "flat", "decoy first"])
    def test_the_overlay_is_found_in_a_folder_in_any_layout(self, doubles, tmp_path, layout):
        folder = tmp_path / "downloads"
        for path in OVERLAY_PATHS:
            name = path.rsplit("/", 1)[-1]
            where = {"tree": folder / path, "clone": folder / "VibeVoice" / path,
                     "flat": folder / name,
                     "decoy first": folder / "zz" / "deeper" / name}[layout]
            where.parent.mkdir(parents=True, exist_ok=True)
            where.write_bytes(_overlay_bytes(path))
        if layout == "decoy first":
            # Another copy of the same size, found first under the package path;
            # only its digest tells it apart, and it is passed over.
            decoy = folder / OVERLAY_PATHS[0]
            decoy.parent.mkdir(parents=True, exist_ok=True)
            real = _overlay_bytes(OVERLAY_PATHS[0])
            decoy.write_bytes(b"#" * len(real))
        patches = tmp_path / "patches"
        vibevoice._adopt_overlay(doubles["overlay"], folder, patches, lambda text: None)
        for path in OVERLAY_PATHS:
            assert (patches / path).read_bytes() == _overlay_bytes(path)

    def test_an_overlay_file_a_folder_lacks_or_has_wrong_is_refused(self, doubles, tmp_path):
        folder = tmp_path / "downloads"
        folder.mkdir()
        for path in OVERLAY_PATHS[1:]:
            (folder / path.rsplit("/", 1)[-1]).write_bytes(_overlay_bytes(path))
        with pytest.raises(vibevoice.VibeVoiceError,
                           match=re.escape(RAW + OVERLAY_PATHS[0])):
            vibevoice._adopt_overlay(doubles["overlay"], folder, tmp_path / "p",
                                     lambda text: None)
        (folder / "configuration_vibevoice.py").write_bytes(b"# the wheel's own copy\n")
        with pytest.raises(vibevoice.VibeVoiceError, match=f"at commit {COMMIT[:12]}"):
            vibevoice._adopt_overlay(doubles["overlay"], folder, tmp_path / "p",
                                     lambda text: None)

    def test_the_manual_sources_list_the_overlay_under_its_own_names(self, doubles):
        found = vibevoice.sources("runtime")
        overlay = [row for row in found if row["url"].startswith(RAW)]
        assert [row["url"] for row in overlay] == [RAW + path for path in OVERLAY_PATHS]
        assert [row["save_as"] for row in overlay] == [path.rsplit("/", 1)[-1]
                                                      for path in OVERLAY_PATHS]


class TestUninstall:
    def test_it_removes_the_runtime_and_models_and_keeps_settings_and_calibration(self, windows):
        _install_runtime_files()
        _install_model_files()
        vibevoice.set_settings({"steps": 20})
        vibevoice.note_peak(MODEL, 123_456_789)
        found = vibevoice.uninstall()
        assert not found.runtime_installed and not found.model_installed
        assert not paths.vibevoice_runtime_root().exists()
        assert not paths.vibevoice_models_root().exists()
        assert vibevoice.settings()["steps"] == 20
        assert vibevoice.calibration()[f"{MODEL}@bf16"]["peak_bytes"] == 123_456_789


# --------------------------------------------------------------------------- #
# The pin tool's solver, against recorded records
# --------------------------------------------------------------------------- #


def _file(filename, digest=None, size=10, yanked=False, requires_python=">=3.9"):
    return {"filename": filename, "packagetype": "bdist_wheel", "yanked": yanked,
            "requires_python": requires_python, "size": size,
            "url": f"https://files.pythonhosted.org/packages/x/{filename}",
            "digests": {"sha256": digest or hashlib.sha256(filename.encode()).hexdigest()}}


def _pure(name, version, **kwargs):
    return _file(f"{name}-{version}-py3-none-any.whl", **kwargs)


def _binary(name, version, tag="cp313-cp313-win_amd64", **kwargs):
    return _file(f"{name}-{version}-{tag}.whl", **kwargs)


def _universe(tool, projects: dict, requires: dict):
    """``projects``: {name: {version: [files]}}; ``requires``: {(name, version): [reqs]}."""

    def project(name):
        return {"info": {"name": name}, "releases": dict(projects[name.casefold()])}

    def release(name, version):
        return {"info": {"name": name, "version": version,
                         "requires_dist": list(requires.get((name.casefold(), version), []))},
                "urls": list(projects[name.casefold()][version])}

    return tool.PyPI(project=project, release=release)


TORCH = {"index": "https://download.pytorch.org/whl/cu128/torch/", "package": "torch",
         "version": "2.8.0"}


class TestThePinToolsSolver:
    def _plain(self, tool):
        projects = {
            "alpha": {"1.0": [_pure("alpha", "1.0")]},
            "beta": {"1.0": [_pure("beta", "1.0")], "2.0": [_pure("beta", "2.0")]},
            "gamma": {"1.0": [_binary("gamma", "1.0")], "2.0": [_binary("gamma", "2.0")],
                      "3.0": [_binary("gamma", "3.0")],
                      "4.0rc1": [_binary("gamma", "4.0rc1")]},
            "theta": {"1.0": [_pure("theta", "1.0")], "1.1": [_pure("theta", "1.1",
                                                                     yanked=True)]},
            "torch": {"2.8.0": [_binary("torch", "2.8.0")]},
        }
        requires = {
            # gamma is declared before beta so that it is chosen (at 3.0) before
            # beta 2.0's ``gamma<2`` arrives, which is what makes this a re-choice.
            ("alpha", "1.0"): ["gamma>=1", "beta", "delta; extra == \"demo\"",
                               "epsilon; platform_system == \"Linux\"", "zeta", "torch"],
            ("beta", "2.0"): ["gamma<2"],
            ("torch", "2.8.0"): ["theta>=1", "nvidia-x; platform_system == \"Linux\""],
        }
        return _universe(tool, projects, requires)

    def test_a_later_constraint_re_chooses_an_earlier_package_downward(self):
        """alpha wants gamma>=1 (newest 3.0 taken first); beta 2.0 then wants
        gamma<2, and gamma is chosen again at 1.0. Extras, other platforms and
        the excluded table are skipped; torch's own declarations take part."""
        tool = _tool()
        said = []
        chosen = tool.solve((("alpha", "==1.0"),), self._plain(tool), {"zeta": "unused"},
                            said.append, torch=TORCH)
        assert {key: choice.version for key, choice in chosen.items()} == {
            "alpha": "1.0", "beta": "2.0", "gamma": "1.0", "theta": "1.0"}
        assert chosen["gamma"].kind == tool.BINARY and chosen["beta"].kind == tool.PURE
        assert any("re-chosen" in line for line in said)
        tool.requirements_met(chosen, {"zeta": "unused"}, "2.8.0",
                              ["theta>=1", "nvidia-x; platform_system == \"Linux\""],
                              roots=(("alpha", "==1.0"),))
        assert [row[0] for row in tool.triples(chosen, roots=(("alpha", "==1.0"),))] == [
            "alpha", "beta", "gamma", "theta"]

    def test_the_newest_root_gives_way_when_it_contradicts_a_pinned_one(self):
        """diffusers 0.40 asks for a hub transformers forbids; the solver bans
        that version and takes 0.39, which is the rule the manifest states."""
        tool = _tool()
        projects = {
            "trans": {"4.51.3": [_pure("trans", "4.51.3")]},
            "diff": {"0.39.0": [_pure("diff", "0.39.0")], "0.40.0": [_pure("diff", "0.40.0")]},
            "hub": {"0.36.2": [_pure("hub", "0.36.2")], "1.30.0": [_pure("hub", "1.30.0")]},
            "torch": {"2.8.0": [_binary("torch", "2.8.0")]},
        }
        requires = {
            ("trans", "4.51.3"): ["hub<1.0,>=0.30.0"],
            ("diff", "0.40.0"): ["hub>=1.23.0,<2.0"],
            ("diff", "0.39.0"): ["hub>=0.34.0,<2.0"],
        }
        said = []
        chosen = tool.solve((("trans", "==4.51.3"), ("diff", "")),
                            _universe(tool, projects, requires), {}, said.append, torch=TORCH)
        assert chosen["diff"].version == "0.39.0"
        assert chosen["hub"].version == "0.36.2"
        assert any("gives way" in line for line in said)

    def test_a_contradiction_between_exact_roots_is_a_refusal(self):
        tool = _tool()
        projects = {
            "a": {"1.0": [_pure("a", "1.0")]}, "b": {"1.0": [_pure("b", "1.0")]},
            "c": {"1.0": [_pure("c", "1.0")], "2.0": [_pure("c", "2.0")]},
            "torch": {"2.8.0": [_binary("torch", "2.8.0")]},
        }
        requires = {("a", "1.0"): ["c<2"], ("b", "1.0"): ["c>=2"]}
        with pytest.raises(tool.PinError, match="no release that satisfies"):
            tool.solve((("a", "==1.0"), ("b", "==1.0")), _universe(tool, projects, requires),
                       {}, lambda text: None, torch=TORCH)

    def test_a_package_with_only_a_free_threaded_wheel_is_refused(self):
        tool = _tool()
        projects = {
            "a": {"1.0": [_pure("a", "1.0")]},
            "t": {"1.0": [_binary("t", "1.0", tag="cp313-cp313t-win_amd64")]},
            "torch": {"2.8.0": [_binary("torch", "2.8.0")]},
        }
        with pytest.raises(tool.PinError, match="no release that satisfies"):
            tool.solve((("a", "==1.0"),), _universe(tool, projects, {("a", "1.0"): ["t"]}), {},
                       lambda text: None, torch=TORCH)

    def test_requirements_met_refuses_a_pin_that_contradicts_another(self):
        tool = _tool()
        good = {"a": tool.Choice("a", "1.0", tool.PURE, {}, ("b>=2",)),
                "b": tool.Choice("b", "2.0", tool.PURE, {}, ())}
        tool.requirements_met(good, {}, "2.8.0", [], roots=(("a", "==1.0"),))
        bad = {"a": tool.Choice("a", "1.0", tool.PURE, {}, ("b>=3",)),
               "b": tool.Choice("b", "2.0", tool.PURE, {}, ())}
        with pytest.raises(tool.PinError, match="needs b>=3"):
            tool.requirements_met(bad, {}, "2.8.0", [], roots=(("a", "==1.0"),))
        missing = {"a": tool.Choice("a", "1.0", tool.PURE, {}, ("c",))}
        with pytest.raises(tool.PinError, match="does not ship it"):
            tool.requirements_met(missing, {}, "2.8.0", [], roots=(("a", "==1.0"),))
        tool.requirements_met(missing, {"c": "not needed"}, "2.8.0", [],
                              roots=(("a", "==1.0"),))
        with pytest.raises(tool.PinError, match="torch is pinned"):
            tool.requirements_met({"a": tool.Choice("a", "1.0", tool.PURE, {}, ("torch>=2.9",))},
                                  {}, "2.8.0", [], roots=(("a", "==1.0"),))
        with pytest.raises(tool.PinError, match="torch 2.8.0 needs"):
            tool.requirements_met(good, {}, "2.8.0", ["sympy>=1.13"], roots=(("a", "==1.0"),))

    @pytest.mark.parametrize("filename, kind", [
        ("numpy-2.5.3-cp313-cp313-win_amd64.whl", "binary"),
        ("tokenizers-0.21.4-cp39-abi3-win_amd64.whl", "abi3"),
        ("safetensors-0.8.0-cp310-abi3-win_amd64.whl", "abi3"),
        ("soundfile-0.13.1-py2.py3-none-win_amd64.whl", "windows"),
        ("tqdm-4.70.1-py3-none-any.whl", "pure"),
        ("colorama-0.4.6-py2.py3-none-any.whl", "pure"),
        ("torch-2.8.0-cp313-cp313t-win_amd64.whl", ""),
        ("numpy-2.5.3-cp312-cp312-win_amd64.whl", ""),
        ("numpy-2.5.3-cp313-cp313-manylinux_2_28_x86_64.whl", ""),
        ("safetensors-0.8.0-cp314-abi3-win_amd64.whl", ""),
        ("not a wheel.txt", ""),
    ])
    def test_wheel_kinds(self, filename, kind):
        assert _tool().classify(filename) == kind

    def test_the_compiled_wheel_is_preferred_and_two_of_a_kind_refused(self):
        tool = _tool()
        files = [_pure("cn", "3.5.1"), _binary("cn", "3.5.1"),
                 _binary("cn", "3.5.1", tag="cp37-abi3-win_amd64")]
        kind, item = tool.usable_wheel(files)
        assert kind == tool.BINARY and item["filename"].endswith("cp313-cp313-win_amd64.whl")
        kind, item = tool.usable_wheel([_pure("x", "1")])
        assert kind == tool.PURE
        assert tool.usable_wheel([_binary("x", "1", tag="cp313-cp313t-win_amd64")]) == ("", None)
        with pytest.raises(tool.PinError, match="more than one"):
            tool.usable_wheel([_binary("x", "1"), _binary("x", "1-1")])

    def test_the_notes_say_what_is_and_is_not_pinned(self):
        tool = _tool()
        state = tool.State(closure_complete=True, model_hashed=False, torch_recorded=False)
        notes = tool.notes_for(state, {"version": "2.8.0"}, {})
        assert notes.startswith("PARTIALLY PINNED")
        assert "DECLARED BUT NOT HASHED" in notes and "torch 2.8.0" in notes
        assert "--torch" in notes and "--model" in notes and "httpx" in notes
        assert state.pinned is False
        done = tool.State(closure_complete=True, model_hashed=True, torch_recorded=True,
                          shards=10, overlay_pinned=True, presets_pinned=True)
        assert done.pinned is True
        assert tool.notes_for(done, {"version": "2.8.0", "sha256": "f" * 64}, {}).startswith(
            "PINNED")

    def test_it_refuses_a_sibling_manifest_and_two_digests_for_one_name(self, tmp_path):
        tool = _tool()
        sibling = tmp_path / "managed-pocket-models.json"
        sibling.write_text("{}", encoding="utf-8")
        assert tool.main(["--manifest", str(sibling), "--check"]) == 2
        with pytest.raises(tool.PinError, match="two digests"):
            tool._committed({"runtime": {"platforms": [{"artifacts": [
                {"filename": "a.whl", "sha256": "a" * 64},
                {"filename": "a.whl", "sha256": "b" * 64}]}]}})

    def test_the_markers_spell_the_machine_the_way_windows_does(self):
        """``AMD64`` in upper case, so hf-xet's lower-case marker is false here
        exactly as it is for pip on the user's machine."""
        tool = _tool()
        assert tool.MARKERS["platform_machine"] == "AMD64"
        assert tool.MARKERS["python_version"] == "3.13"
        assert tool.MARKERS["platform_system"] == "Windows"
        from packaging.requirements import Requirement

        assert tool._applies(Requirement(
            'hf-xet>=1.1.3; platform_machine == "x86_64" or platform_machine == "amd64"')) \
            is False
        assert tool._applies(Requirement('colorama; platform_system == "Windows"')) is True
        assert tool._applies(Requirement('setuptools; python_version >= "3.12"')) is True
        assert tool._applies(Requirement('accelerate; extra == "accelerate"')) is False


# --------------------------------------------------------------------------- #
# Phase 4: the overlay, the new wheels, the Realtime model and its presets
# --------------------------------------------------------------------------- #


class TestTheManifestsPhaseFourClaims:
    def test_the_overlay_is_the_five_files_at_the_commit_pinned_here(self):
        """Code written over the unpacked package, so every file is pinned by size
        and SHA-256 -- the bytes a clone of the commit has and raw.githubusercontent.com
        served -- and says why it is there."""
        found = vibevoice.manifest(refresh=True)["runtime"]["overlay"]
        assert [item["path"] for item in found] == list(OVERLAY_PATHS)
        for item in found:
            size, digest = OVERLAY_PINS[item["path"]]
            assert item["bytes"] == size and item["sha256"] == digest, item["path"]
            assert item["url"] == RAW + item["path"]
            assert item["source"] == {"repository": "https://github.com/microsoft/VibeVoice",
                                      "commit": COMMIT}
            assert len(item["why"]) > 30
        read = vibevoice.overlay_files()
        assert [item.path for item in read] == list(OVERLAY_PATHS)
        assert all(item.artifact.pinned and item.commit == COMMIT for item in read)
        assert "_convert_dtype_to_string" in found[0]["why"]

    def test_bitsandbytes_and_peft_join_the_closure_pinned_exactly(self):
        found = vibevoice.manifest(refresh=True)
        rows = {_normalise(name): (version, kind)
                for name, version, kind in found["runtime"]["closure"]}
        assert rows["bitsandbytes"] == ("0.48.2", "windows")
        assert rows["peft"] == ("0.17.1", "pure")
        wheels = {item["filename"]: item for item in found["runtime"]["platforms"][0]["artifacts"]
                  if item.get("filename")}
        bnb = wheels["bitsandbytes-0.48.2-py3-none-win_amd64.whl"]
        assert bnb["bytes"] == 58992447
        assert bnb["sha256"] == "a048c285eb6ff53a8d189880e9dfa421d2bfb54e8cab263311757cf5b742d865"
        peft = wheels["peft-0.17.1-py3-none-any.whl"]
        assert peft["bytes"] == 504896
        assert peft["sha256"] == "3d129d64def3d74779c32a080d2567e5f7b674e77d546e3585138216d903f99e"
        tool = _tool()
        assert ("bitsandbytes", "==0.48.2") in tool.ROOTS and ("peft", "==0.17.1") in tool.ROOTS
        assert tool.classify(bnb["filename"]) == tool.WINDOWS

    def test_the_realtime_model_is_declared_and_its_presets_are_pinned(self):
        entry = vibevoice.manifest(refresh=True)["models"][REALTIME]
        assert entry["label"] == "VibeVoice Realtime 0.5B" and entry["kind"] == "realtime"
        assert entry["repo"] == "microsoft/VibeVoice-Realtime-0.5B"
        assert entry["revision"] == "main" and entry["license"] == "MIT"
        files = {item["filename"]: item for item in entry["files"]}
        assert list(files) == ["config.json", "preprocessor_config.json",
                               "model.safetensors.index.json"]
        assert not files["config.json"].get("optional")
        assert files["preprocessor_config.json"]["optional"] is True
        assert files["preprocessor_config.json"]["local_name"] == paths.VIBEVOICE_UPSTREAM_CONFIG
        assert files["model.safetensors.index.json"]["optional"] is True
        assert entry["single_weights"]["filename"] == "model.safetensors"
        for item in list(files.values()) + [entry["single_weights"]]:
            assert item["sha256"] is None and item["bytes"] is None, "declared, not hashed"
            assert item["url"].startswith(
                "https://huggingface.co/microsoft/VibeVoice-Realtime-0.5B/resolve/main/")
        tokenizer = entry["tokenizer"]
        assert tokenizer["repo"] == "Qwen/Qwen2.5-0.5B"
        assert tokenizer["dirname"] == paths.VIBEVOICE_REALTIME_TOKENIZER_DIRNAME
        assert "qwen" in tokenizer["dirname"]
        assert [item["filename"] for item in tokenizer["files"]] == list(TOKENIZER_FILES)
        assert entry["estimates"]["weights_bytes"] == 2_000_000_000
        assert entry["estimates"]["working_bytes"] == 1024 ** 3
        assert entry["estimates"]["ram_bytes"] == 2 * 1024 ** 3
        assert "approximate until pinned" in entry["estimates"]["weights_note"]
        assert entry["defaults"] == {"steps": 5, "cfg_scale": 1.5}
        assert entry["precisions"] == ["bf16"]
        assert entry["lora"] is False and entry["max_speakers"] == 1
        assert entry["local_config"]["processor_class"] == "VibeVoiceStreamingProcessor"
        assert entry["voices_source"] == {"repository": "https://github.com/microsoft/VibeVoice",
                                          "commit": COMMIT,
                                          "path": "demo/voices/streaming_model"}

    def test_the_twenty_five_presets_carry_what_their_names_say(self):
        tool = _tool()
        voices = vibevoice.manifest(refresh=True)["models"][REALTIME]["voices"]
        assert [item["id"] for item in voices] == list(tool.PRESETS)
        assert len(voices) == 25
        for item in voices:
            assert re.fullmatch(r"[A-Za-z0-9_-]+", item["id"])
            assert item["url"] == f"{RAW}demo/voices/streaming_model/{item['id']}.pt"
            assert re.fullmatch(r"[0-9a-f]{64}", item["sha256"]) and item["bytes"] > 1_000_000
            meta = tool.preset_meta(item["id"])
            assert {key: item[key] for key in meta} == meta, item["id"]
            assert item["experimental"] is (item["language"] != "en")
        carter = voices[0]
        assert (carter["name"], carter["language"], carter["gender"]) == ("Carter", "en", "man")
        assert carter["sha256"] == \
            "a7bfdf1cd4939c22469bcfc6f427ae9c4467b3df46c2c14303a39c294cfc6897"
        samuel = next(item for item in voices if item["id"] == "in-Samuel_man")
        assert (samuel["language"], samuel["accent"], samuel["experimental"]) == \
            ("en", "in", False), "released with the English voices, not the experimental ones"
        german = next(item for item in voices if item["id"] == "de-Spk1_woman")
        assert (german["name"], german["language"], german["experimental"]) == \
            ("Speaker 1", "de", True)
        assert [item["language"] for item in voices[:7]] == ["en"] * 7, "English first"

    def test_the_7b_gains_its_kind_precisions_and_lora(self):
        entry = vibevoice.manifest(refresh=True)["models"][MODEL]
        assert entry["kind"] == "longform"
        assert entry["defaults"] == {"steps": 10, "cfg_scale": 1.3}
        assert entry["precisions"] == ["bf16", "int8", "nf4"]
        assert entry["lora"] is True and entry["max_speakers"] == 4
        assert entry["estimates"]["weights_by_precision"] == {
            "bf16": 18_700_000_000, "int8": 11_500_000_000, "nf4": 7_500_000_000}
        assert "voices" not in entry

    def test_the_notes_say_what_is_hashed_and_what_is_declared(self):
        found = vibevoice.manifest(refresh=True)
        notes = found["notes"]
        assert found["pinned"] is False and found["version"] >= 3
        assert notes.startswith("PARTIALLY PINNED")
        assert "bitsandbytes 0.48.2" in notes and "libbitsandbytes_cuda128.dll" in notes
        assert "peft 0.17.1" in notes
        assert "HASHED as well: the runtime overlay" in notes and COMMIT[:12] in notes
        assert "25 preset voices" in notes
        assert "DECLARED BUT NOT HASHED" in notes
        assert "VibeVoice Realtime 0.5B (microsoft/VibeVoice-Realtime-0.5B)" in notes
        assert "Qwen/Qwen2.5-0.5B tokenizer" in notes and "Qwen/Qwen2.5-7B tokenizer" in notes

    def test_a_manifest_overlay_that_is_not_pinned_or_leaves_the_package_is_refused(
            self, windows, monkeypatch):
        """Code with no committed digest is a broken manifest, not something to
        check against a host that states no digest; and an overlay path is held to
        one module inside the package."""
        base = json.loads(json.dumps(vibevoice.manifest(refresh=True)))
        for change, words in [
            ({"sha256": None}, "does not pin"),
            ({"bytes": 0}, "does not pin"),
            ({"path": "../sitecustomize.py"}, "not one module inside"),
            ({"path": "torch/__init__.py"}, "not one module inside"),
            ({"path": "vibevoice/../evil.py"}, "not one module inside"),
            ({"path": "vibevoice/modular/data.json"}, "not one module inside"),
            ({"url": "http://raw.githubusercontent.com/x.py"}, "HTTPS"),
        ]:
            found = json.loads(json.dumps(base))
            found["runtime"]["overlay"][1].update(change)
            monkeypatch.setattr(vibevoice, "manifest", lambda refresh=False, f=found: f)
            with pytest.raises(vibevoice.VibeVoiceError, match=words):
                vibevoice.overlay_files()
            assert vibevoice.pinned() is False, change
        found = json.loads(json.dumps(base))
        found["runtime"]["overlay"] = []
        monkeypatch.setattr(vibevoice, "manifest", lambda refresh=False: found)
        assert vibevoice.pinned() is False, "a runtime with no overlay cannot run the Realtime model"
        with pytest.raises(vibevoice.VibeVoiceError, match="overlay"):
            vibevoice.install_runtime()

    def test_a_preset_whose_id_is_not_a_plain_name_is_refused(self, monkeypatch):
        found = json.loads(json.dumps(vibevoice.manifest(refresh=True)))
        found["models"][REALTIME]["voices"][0]["id"] = "../en-Carter_man"
        monkeypatch.setattr(vibevoice, "manifest", lambda refresh=False: found)
        with pytest.raises(vibevoice.VibeVoiceError, match="not a plain file name"):
            vibevoice.bundle(REALTIME)
        found["models"][REALTIME]["voices"][0]["id"] = "en-Davis_man"
        with pytest.raises(vibevoice.VibeVoiceError, match="twice"):
            vibevoice.bundle(REALTIME)

    def test_a_manifest_cannot_give_the_realtime_model_a_lora_or_more_voices_than_four(
            self, monkeypatch):
        found = json.loads(json.dumps(vibevoice.manifest(refresh=True)))
        found["models"][REALTIME]["lora"] = True
        found["models"][MODEL]["max_speakers"] = 9
        found["models"][MODEL]["defaults"] = {"steps": 0, "cfg_scale": 9}
        monkeypatch.setattr(vibevoice, "manifest", lambda refresh=False: found)
        assert vibevoice.bundle(REALTIME).lora is False, "LoRA is the long-form model's"
        assert vibevoice.bundle(MODEL).max_speakers == 4
        assert vibevoice.bundle(MODEL).defaults == {"steps": 10, "cfg_scale": 1.3}

    def test_a_kind_or_precision_this_build_does_not_know_is_refused(self, monkeypatch):
        base = json.loads(json.dumps(vibevoice.manifest(refresh=True)))
        found = json.loads(json.dumps(base))
        found["models"][MODEL]["kind"] = "karaoke"
        monkeypatch.setattr(vibevoice, "manifest", lambda refresh=False: found)
        with pytest.raises(vibevoice.VibeVoiceError, match="does not know how to run"):
            vibevoice.bundle(MODEL)
        found = json.loads(json.dumps(base))
        found["models"][MODEL]["precisions"] = ["bf16", "fp8"]
        monkeypatch.setattr(vibevoice, "manifest", lambda refresh=False: found)
        with pytest.raises(vibevoice.VibeVoiceError, match="precision"):
            vibevoice.bundle(MODEL)


def _voice_file(path: Path, size: int) -> None:
    """A preset voice's file at its pinned size, sparse after its zip header: status
    reads sizes, and twenty-five real voices would be a hundred megabytes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as handle:
        handle.write(b"PK\x03\x04")
        handle.truncate(size)


def _install_realtime_files(voices=None, tokenizer=True, local_config=True) -> Path:
    """The Realtime model as an install would leave it; ``voices`` the ids present."""
    root = paths.vibevoice_model_root(REALTIME)
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text(json.dumps({"model_type": "vibevoice_streaming"}))
    (root / "model.safetensors").write_bytes(_safetensors())
    if tokenizer:
        folder = root / paths.VIBEVOICE_REALTIME_TOKENIZER_DIRNAME
        folder.mkdir(exist_ok=True)
        for name in TOKENIZER_FILES:
            (folder / name).write_bytes(_files()[name])
    folder = root / paths.VIBEVOICE_VOICES_DIRNAME
    shutil_rmtree(folder)
    for item in vibevoice.bundle(REALTIME).voices:
        if voices is None or item.identifier in voices:
            _voice_file(folder / item.filename, int(item.artifact.size))
    vibevoice._write_json(root / paths.INSTALLED_FILENAME, {
        "schema": 1, "id": REALTIME, "kind": "realtime", "shards": ["model.safetensors"],
        "weights": "single", "bytes": {"model.safetensors": len(_safetensors())}})
    if local_config:
        vibevoice._write_json(root / paths.VIBEVOICE_LOCAL_CONFIG, {
            "processor_class": "VibeVoiceStreamingProcessor",
            "language_model_pretrained_name": str(
                root / paths.VIBEVOICE_REALTIME_TOKENIZER_DIRNAME)})
    return root


def shutil_rmtree(folder: Path) -> None:
    import shutil

    shutil.rmtree(folder, ignore_errors=True)


class TestWhatEachModelIs:
    KEYS = {"id", "label", "kind", "installed", "runtime_installed", "precisions", "lora",
            "max_speakers", "voices", "defaults", "need_vram_bytes", "need_ram_bytes",
            "presets", "message", "download_bytes"}

    def test_models_info_describes_every_model_in_manifest_order(self, windows):
        found = vibevoice.models_info()
        assert [item["id"] for item in found] == [MODEL, REALTIME]
        for item in found:
            assert self.KEYS <= set(item), self.KEYS - set(item)
            json.dumps(item)
        assert vibevoice.MODEL_7B == vibevoice.MODEL_DEFAULT == MODEL
        assert vibevoice.MODEL_REALTIME == REALTIME
        assert vibevoice.PRECISIONS == ("bf16", "int8", "nf4")
        assert vibevoice.PRECISION_LABELS == {"bf16": "Full (bf16)", "int8": "8-bit",
                                              "nf4": "4-bit (NF4)"}

    def test_the_7b_clones_from_samples_in_three_precisions(self, windows):
        info = vibevoice.model_info(MODEL)
        assert info["kind"] == vibevoice.KIND_LONGFORM == "longform"
        assert info["voices"] == "samples" and info["presets"] == []
        assert info["precisions"] == ["bf16", "int8", "nf4"]
        assert info["lora"] is True and info["max_speakers"] == 4
        assert info["defaults"] == {"steps": 10, "cfg_scale": 1.3}
        working = 2 * 1024 ** 3
        assert info["need_vram_bytes"] == {"bf16": 18_700_000_000 + working,
                                           "int8": 11_500_000_000 + working,
                                           "nf4": 7_500_000_000 + working}
        assert info["need_ram_bytes"] == 3 * 1024 ** 3
        assert (info["installed"], info["runtime_installed"], info["ready"]) == (False,) * 3
        assert info["message"] == "Neither VibeVoice's runtime nor VibeVoice 7B is installed."
        assert info["download_bytes"] == vibevoice.bundle(MODEL).download_bytes

    def test_the_realtime_model_speaks_with_presets_in_full_precision_only(self, windows):
        info = vibevoice.model_info(REALTIME)
        assert info["kind"] == vibevoice.KIND_REALTIME == "realtime"
        assert info["voices"] == "presets"
        assert info["precisions"] == ["bf16"] and info["lora"] is False
        assert info["max_speakers"] == 1
        assert info["defaults"] == {"steps": 5, "cfg_scale": 1.5}
        assert info["need_vram_bytes"] == {"bf16": 2_000_000_000 + 1024 ** 3}
        assert info["need_ram_bytes"] == 2 * 1024 ** 3
        assert len(info["presets"]) == 25
        assert not any(item["installed"] for item in info["presets"])
        voices = sum(item.artifact.size for item in vibevoice.bundle(REALTIME).voices)
        assert info["download_bytes"] == vibevoice.bundle(REALTIME).download_bytes
        assert info["download_bytes"] >= 2_000_000_000 + voices

    def test_blank_is_the_settings_model_and_an_unknown_one_is_refused(self, windows):
        assert vibevoice.model_info()["id"] == MODEL
        vibevoice.set_settings({"model_id": REALTIME})
        assert vibevoice.model_info("")["id"] == REALTIME
        with pytest.raises(vibevoice.VibeVoiceError, match="not a VibeVoice model"):
            vibevoice.model_info("vibevoice-1.5b")

    def test_the_7bs_message_names_exactly_what_cloning_is_missing(self, windows):
        """What Voice Chat's clone form reads to say whether cloning is ready."""
        _install_runtime_files()
        assert vibevoice.model_info(MODEL)["message"] == "VibeVoice 7B is not installed."
        root = _install_model_files(tokenizer=False)
        info = vibevoice.model_info(MODEL)
        assert info["message"] == ("VibeVoice 7B's tokenizer is missing and the model has to "
                                   "be installed again.")
        assert info["installed"] is False and info["ready"] is False
        _install_model_files()
        (root / SHARDS[1]).unlink()
        assert vibevoice.model_info(MODEL)["message"] == (
            f"VibeVoice 7B is missing {SHARDS[1]} and has to be installed again.")
        _install_model_files()
        info = vibevoice.model_info(MODEL)
        assert info["message"] == "" and info["installed"] and info["ready"]
        assert info["runtime_installed"] and info["download_bytes"] == 0
        _install_runtime_files(closure="0123456789abcdef")
        info = vibevoice.model_info(MODEL)
        assert info["message"] == ("VibeVoice's runtime was installed by an earlier build and "
                                   "has to be installed again.")
        assert info["installed"] is True and info["ready"] is False

    def test_the_realtime_models_message_counts_its_presets(self, windows):
        _install_runtime_files()
        root = _install_realtime_files(voices=())
        info = vibevoice.model_info(REALTIME)
        assert info["installed"] is True and info["ready"] is False
        assert info["message"] == ("None of VibeVoice Realtime 0.5B's 25 preset voices are "
                                   "installed, so the model has to be installed again.")
        assert info["download_bytes"] == sum(item.artifact.size
                                             for item in vibevoice.bundle(REALTIME).voices)
        _install_realtime_files(voices=("en-Carter_man", "de-Spk0_man"))
        info = vibevoice.model_info(REALTIME)
        assert info["ready"] is True and info["presets_installed"] == 2
        assert info["message"] == ("23 of VibeVoice Realtime 0.5B's 25 preset voices are "
                                   "missing (installing the model again fetches them).")
        _install_realtime_files()
        info = vibevoice.model_info(REALTIME)
        assert info["message"] == "" and info["presets_installed"] == 25
        assert info["download_bytes"] == 0
        with open(root / "voices" / "en-Emma_woman.pt", "r+b") as handle:
            handle.truncate(100)
        info = vibevoice.model_info(REALTIME)
        assert info["presets_installed"] == 24, "a truncated voice is not an installed one"
        assert info["message"].startswith("1 of VibeVoice Realtime 0.5B's 25 preset voices is "
                                          "missing")

    def test_status_of_the_realtime_model_says_one_weights_file_and_its_voices(self, windows):
        _install_runtime_files()
        _install_realtime_files(voices=())
        found = vibevoice.status(REALTIME)
        assert found.model_installed and found.tokenizer_installed and not found.ready
        assert (found.presets_installed, found.presets_total) == (0, 25)
        assert found.message == ("Setup required — VibeVoice's preset voices (install the "
                                 "model again) still to install.")
        assert "25 of its 25 preset voices are missing" in found.model_message
        _install_realtime_files()
        found = vibevoice.status(REALTIME)
        assert found.ready and found.message == "Installed."
        assert found.model_message.startswith("Installed — VibeVoice Realtime 0.5B, one "
                                              "weights file")
        assert "25 of 25 preset voices" in found.model_message
        assert vibevoice.status(MODEL).model_message.startswith("Not installed — VibeVoice 7B")

    def test_public_status_carries_every_model_and_the_library(self, windows, tmp_path):
        lora = vibevoice.add_lora(str(_adapter(tmp_path / "voice-lora")))
        found = vibevoice.public_status()
        assert [item["id"] for item in found["models"]] == [MODEL, REALTIME]
        assert found["models"][1]["voices"] == "presets"
        assert found["loras"] == [lora]
        assert found["precision_labels"] == vibevoice.PRECISION_LABELS
        json.dumps(found)


class TestPresets:
    ENGLISH = ["en-Carter_man", "en-Davis_man", "en-Emma_woman", "en-Frank_man",
               "en-Grace_woman", "en-Mike_man", "in-Samuel_man"]

    def test_they_are_listed_english_first_with_their_labels(self, windows):
        found = vibevoice.presets()
        assert [item["id"] for item in found][:7] == self.ENGLISH
        labels = {item["id"]: item["language_label"] for item in found}
        assert labels["en-Carter_man"] == "English"
        assert labels["in-Samuel_man"] == "English (India)"
        assert (labels["de-Spk0_man"], labels["jp-Spk1_woman"], labels["kr-Spk1_man"],
                labels["sp-Spk0_woman"], labels["nl-Spk0_man"]) == (
            "German", "Japanese", "Korean", "Spanish", "Dutch")
        for item in found:
            assert set(item) == {"id", "name", "language", "language_label", "accent",
                                 "gender", "experimental", "installed", "bytes"}
            assert item["experimental"] is (item["language"] != "en")
        assert sum(1 for item in found if item["experimental"]) == 18
        assert vibevoice.presets(MODEL) == [], "the 7B's voices are Voice Box samples"
        with pytest.raises(vibevoice.VibeVoiceError):
            vibevoice.presets("vibevoice-1.5b")

    def test_installed_follows_the_file_at_its_pinned_size(self, windows):
        root = _install_realtime_files(voices=("en-Grace_woman",))
        flags = {item["id"]: item["installed"] for item in vibevoice.presets()}
        assert flags["en-Grace_woman"] is True and sum(flags.values()) == 1
        with open(root / "voices" / "en-Grace_woman.pt", "ab") as handle:
            handle.write(b"x")
        assert not any(item["installed"] for item in vibevoice.presets())

    def test_a_voice_path_is_only_ever_made_from_an_id_the_manifest_names(self, windows):
        folder = vibevoice.voices_dir()
        assert folder == paths.vibevoice_model_root(REALTIME) / "voices"
        assert vibevoice.voices_dir(REALTIME) == folder
        assert vibevoice.preset_path("en-Carter_man") == folder / "en-Carter_man.pt"
        assert vibevoice.preset_path("in-Samuel_man", REALTIME) == folder / "in-Samuel_man.pt"
        for stem in ("../en-Carter_man", "en-Carter_man.pt", "EN-CARTER_MAN", "nobody", "",
                     "/etc/passwd", "en-Carter_man/../../x", None):
            with pytest.raises(vibevoice.VibeVoiceError, match="preset voices"):
                vibevoice.preset_path(stem)
        with pytest.raises(vibevoice.VibeVoiceError, match="has no preset voices"):
            vibevoice.preset_path("en-Carter_man", MODEL)
        with pytest.raises(vibevoice.VibeVoiceError):
            vibevoice.voices_dir("../elsewhere")


class TestRefusalsPerModel:
    def test_each_model_is_refused_for_what_it_is_missing(self, windows):
        assert vibevoice.refusal(model_id=REALTIME) == \
            "VibeVoice's runtime is not installed — install it below."
        _install_runtime_files()
        assert vibevoice.refusal(model_id=REALTIME) == \
            "The VibeVoice Realtime 0.5B model is not installed — install it below."
        _install_realtime_files(voices=())
        assert vibevoice.refusal(model_id=REALTIME) == (
            "None of the VibeVoice Realtime 0.5B model's preset voices are installed — "
            "install the model again.")
        _install_realtime_files(tokenizer=False, voices=("en-Carter_man",))
        shutil_rmtree(paths.vibevoice_model_root(REALTIME)
                      / paths.VIBEVOICE_REALTIME_TOKENIZER_DIRNAME)
        assert vibevoice.refusal(model_id=REALTIME) == (
            "The VibeVoice Realtime 0.5B model's tokenizer is not installed — install the "
            "model again.")
        _install_realtime_files(voices=("en-Carter_man",))
        assert vibevoice.refusal(model_id=REALTIME) == ""
        assert vibevoice.refusal(model_id=MODEL) == \
            "The VibeVoice 7B model is not installed — install it below.", "the other model"
        assert vibevoice.refusal() == vibevoice.refusal(model_id=MODEL), "blank: the settings'"
        vibevoice.set_settings({"model_id": REALTIME})
        assert vibevoice.refusal() == ""

    def test_a_missing_file_a_stale_runtime_and_an_unknown_model_are_named(self, windows):
        _install_runtime_files()
        root = _install_model_files()
        (root / SHARDS[0]).unlink()
        assert vibevoice.refusal(model_id=MODEL) == \
            f"The VibeVoice 7B model is missing {SHARDS[0]} — install it again below."
        _install_runtime_files(closure="0123456789abcdef")
        assert vibevoice.refusal(model_id=MODEL) == (
            "VibeVoice's runtime was installed by an earlier build and has to be installed "
            "again — install it below.")
        assert vibevoice.refusal(model_id="vibevoice-1.5b") == \
            "'vibevoice-1.5b' is not a VibeVoice model this build knows."


class TestWhatARenderNeedsPerPrecision:
    WORKING = 2 * 1024 ** 3

    def test_each_precision_has_its_own_estimate(self, voice_root):
        assert vibevoice.need_vram_bytes(MODEL, "bf16") == 18_700_000_000 + self.WORKING
        assert vibevoice.need_vram_bytes(MODEL, "int8") == 11_500_000_000 + self.WORKING
        assert vibevoice.need_vram_bytes(MODEL, "nf4") == 7_500_000_000 + self.WORKING
        assert vibevoice.need_vram_bytes(REALTIME, "bf16") == 2_000_000_000 + 1024 ** 3
        assert vibevoice.need_ram_bytes(REALTIME, "bf16") == 2 * 1024 ** 3
        with pytest.raises(vibevoice.VibeVoiceError, match=r"Full \(bf16\) only"):
            vibevoice.need_vram_bytes(REALTIME, "int8")
        with pytest.raises(vibevoice.VibeVoiceError):
            vibevoice.need_ram_bytes(REALTIME, "nf4")
        with pytest.raises(vibevoice.VibeVoiceError):
            vibevoice.need_vram_bytes(MODEL, "fp8")

    def test_blank_is_the_settings_precision_where_the_model_runs_in_it(self, voice_root):
        vibevoice.set_settings({"precision": "nf4"})
        assert vibevoice.need_vram_bytes() == vibevoice.need_vram_bytes(MODEL, "nf4")
        assert vibevoice.need_vram_bytes(MODEL) == vibevoice.need_vram_bytes(MODEL, "nf4")
        assert vibevoice.need_vram_bytes(REALTIME) == vibevoice.need_vram_bytes(REALTIME, "bf16")

    def test_a_peak_is_recorded_and_read_per_precision(self, voice_root):
        vibevoice.note_peak(MODEL, 30_000_000_000, 6_000_000_000, "nf4")
        assert vibevoice.need_vram_bytes(MODEL, "nf4") == 30_000_000_000
        assert vibevoice.need_ram_bytes(MODEL, "nf4") == 6_000_000_000
        assert vibevoice.need_vram_bytes(MODEL, "int8") == 11_500_000_000 + self.WORKING
        assert vibevoice.need_vram_bytes(MODEL, "bf16") == 18_700_000_000 + self.WORKING
        assert set(vibevoice.calibration()) == {f"{MODEL}@nf4"}
        vibevoice.note_peak(MODEL, 5, 0, "fp8")
        assert set(vibevoice.calibration()) == {f"{MODEL}@nf4"}, "an unknown precision is ignored"
        vibevoice.note_peak(REALTIME, 4_000_000_000)
        assert vibevoice.calibration()[f"{REALTIME}@bf16"]["peak_bytes"] == 4_000_000_000
        assert vibevoice.calibration_key(MODEL) == f"{MODEL}@bf16"
        assert vibevoice.calibration_key(MODEL, "NF4") == f"{MODEL}@nf4"
        info = vibevoice.model_info(MODEL)
        assert info["need_vram_bytes"]["nf4"] == 30_000_000_000
        assert info["need_ram_bytes"] == 6_000_000_000, "the most any precision needs"

    def test_a_peak_an_earlier_build_recorded_counts_as_bf16(self, voice_root):
        vibevoice._write_json(paths.vibevoice_calibration_path(), {"schema": 1, "models": {
            MODEL: {"peak_bytes": 40_000_000_000, "rss_bytes": 9_000_000_000, "renders": 3}}})
        assert vibevoice.need_vram_bytes(MODEL, "bf16") == 40_000_000_000
        assert vibevoice.need_ram_bytes(MODEL, "bf16") == 9_000_000_000
        assert vibevoice.need_vram_bytes(MODEL, "int8") == 11_500_000_000 + self.WORKING

    def test_a_precision_without_a_figure_is_a_share_of_the_bf16_weights(self, voice_root,
                                                                         monkeypatch):
        found = json.loads(json.dumps(vibevoice.manifest(refresh=True)))
        del found["models"][MODEL]["estimates"]["weights_by_precision"]["int8"]
        monkeypatch.setattr(vibevoice, "manifest", lambda refresh=False: found)
        entry = vibevoice.bundle(MODEL)
        assert vibevoice._weights_bytes(entry, "int8") == int(
            18_700_000_000 * vibevoice.QUANTISED_SHARE["int8"])


def _adapter(folder: Path, peft_type="LORA", weights="safetensors", rank=8) -> Path:
    """A PEFT adapter pair as training writes one."""
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "adapter_config.json").write_text(json.dumps({
        "peft_type": peft_type, "r": rank, "lora_alpha": 16,
        "target_modules": ["q_proj", "v_proj"],
        "base_model_name_or_path": "Qwen/Qwen2.5-7B"}), encoding="utf-8")
    if weights == "safetensors":
        (folder / "adapter_model.safetensors").write_bytes(_safetensors())
    elif weights == "bin":
        (folder / "adapter_model.bin").write_bytes(b"PK\x03\x04" + b"\0" * 64)
    return folder


def _library() -> list:
    root = paths.vibevoice_loras_root()
    return sorted(child.name for child in root.iterdir()) if root.is_dir() else []


class TestSettingsForPrecisionAndLora:
    def test_the_precision_is_one_the_model_runs_in(self, voice_root):
        assert vibevoice.set_settings({"precision": "int8"})["precision"] == "int8"
        with pytest.raises(vibevoice.VibeVoiceError,
                           match=r"VibeVoice Realtime 0.5B runs in Full \(bf16\) only"):
            vibevoice.set_settings({"model_id": REALTIME, "precision": "nf4"})
        assert vibevoice.settings()["model_id"] == MODEL, "a refused change wrote nothing"
        with pytest.raises(vibevoice.VibeVoiceError, match="Precision is"):
            vibevoice.set_settings({"precision": "fp8"})

    def test_a_choice_for_the_7b_survives_a_trip_to_the_realtime_model(self, voice_root,
                                                                       tmp_path):
        lora = vibevoice.add_lora(str(_adapter(tmp_path / "warm")))
        vibevoice.set_settings({"precision": "nf4", "lora_id": lora["id"], "lora_scale": 0.5})
        found = vibevoice.set_settings({"model_id": REALTIME})
        assert (found["precision"], found["lora_id"], found["lora_scale"]) == ("bf16", "", 0.5)
        with pytest.raises(vibevoice.VibeVoiceError, match="takes no LoRA"):
            vibevoice.set_settings({"lora_id": lora["id"]})
        found = vibevoice.set_settings({"model_id": MODEL})
        assert (found["precision"], found["lora_id"]) == ("nf4", lora["id"])

    @pytest.mark.parametrize("values, words", [
        ({"lora_id": "nothere"}, "not in the library"),
        ({"lora_id": "../loras"}, "not in the library"),
        ({"lora_scale": -0.01}, "0.0 to 2.0"),
        ({"lora_scale": 2.01}, "0.0 to 2.0"),
        ({"lora_scale": "strong"}, "0.0 to 2.0"),
        ({"lora_scale": True}, "0.0 to 2.0"),
        ({"chat": "loud"}, "a set of named values"),
        ({"chat": {"volume": 1}}, "not one of Voice Chat's"),
        ({"chat": {"precision": "fp8"}}, "Precision is"),
        ({"chat": {"lora_scale": 3}}, "0.0 to 2.0"),
        ({"chat": {"lora_id": "nothere"}}, "not in the library"),
        ({"chat": {"card_uuid": "GPU 1; rm"}}, "not a card identifier"),
    ])
    def test_new_values_out_of_range_are_refused(self, voice_root, values, words):
        with pytest.raises(vibevoice.VibeVoiceError, match=re.escape(words)):
            vibevoice.set_settings(values)
        assert not paths.vibevoice_settings_path().exists(), "a refused change wrote nothing"
        assert vibevoice.settings() == {**vibevoice.SETTINGS_DEFAULTS,
                                        "chat": vibevoice.CHAT_DEFAULTS}

    def test_voice_chats_precision_is_one_some_model_runs_in(self, voice_root, monkeypatch):
        found = json.loads(json.dumps(vibevoice.manifest(refresh=True)))
        found["models"][MODEL]["precisions"] = ["bf16", "nf4"]
        monkeypatch.setattr(vibevoice, "manifest", lambda refresh=False: found)
        with pytest.raises(vibevoice.VibeVoiceError, match="No VibeVoice model here runs in 8-bit"):
            vibevoice.set_settings({"chat": {"precision": "int8"}})
        assert vibevoice.set_settings({"chat": {"precision": "nf4"}})["chat"]["precision"] == "nf4"

    def test_the_strength_bounds_are_inclusive(self, voice_root):
        assert vibevoice.set_settings({"lora_scale": 0})["lora_scale"] == 0.0
        assert vibevoice.set_settings({"lora_scale": "2"})["lora_scale"] == 2.0

    def test_the_chat_block_merges_and_is_validated_like_the_rest(self, voice_root, tmp_path):
        found = vibevoice.set_settings({"chat": {"card_uuid": "GPU-aa-bb", "precision": "int8"}})
        assert found["chat"] == {"card_uuid": "GPU-aa-bb", "precision": "int8", "lora_id": "",
                                 "lora_scale": 1.0}
        lora = vibevoice.add_lora(str(_adapter(tmp_path / "chatty")))
        found = vibevoice.set_settings({"chat": {"lora_id": lora["id"], "lora_scale": 1.25}})
        assert found["chat"] == {"card_uuid": "GPU-aa-bb", "precision": "int8",
                                 "lora_id": lora["id"], "lora_scale": 1.25}, "merged"
        assert found["precision"] == "bf16" and found["lora_id"] == "", "the chat's own"
        stored = json.loads(paths.vibevoice_settings_path().read_text(encoding="utf-8"))
        assert stored["chat"]["card_uuid"] == "GPU-aa-bb"

    def test_a_stored_value_that_no_longer_validates_falls_back_alone(self, voice_root):
        vibevoice._write_json(paths.vibevoice_settings_path(), {
            "chat": {"precision": "fp8", "lora_scale": 1.5, "lora_id": "gone", "card_uuid": 7},
            "precision": "int4", "lora_id": "gone", "lora_scale": 9})
        found = vibevoice.settings()
        assert found["chat"] == {"card_uuid": "7", "precision": "bf16", "lora_id": "",
                                 "lora_scale": 1.5}
        assert (found["precision"], found["lora_id"], found["lora_scale"]) == ("bf16", "", 1.0)
        vibevoice._write_json(paths.vibevoice_settings_path(), {"chat": "not a block"})
        assert vibevoice.settings()["chat"] == vibevoice.CHAT_DEFAULTS

    def test_the_defaults_are_never_shared_with_what_settings_answers(self, voice_root):
        vibevoice.settings()["chat"]["precision"] = "nf4"
        assert vibevoice.SETTINGS_DEFAULTS["chat"]["precision"] == "bf16"
        assert vibevoice.settings()["chat"]["precision"] == "bf16"


class TestTheLoraLibrary:
    def test_an_adapter_at_the_folders_root_is_copied_with_its_record(self, voice_root,
                                                                      tmp_path):
        source = _adapter(tmp_path / "my-voice-lora")
        found = vibevoice.add_lora(str(source))
        assert re.fullmatch(r"[0-9a-f]{12}", found["id"])
        assert (found["name"], found["base"], found["parts"]) == ("my-voice-lora", MODEL,
                                                                  ["llm"])
        assert found["created"] and found["bytes"] > 0
        assert found["adapter"] == {"base_model_name_or_path": "Qwen/Qwen2.5-7B", "r": 8,
                                    "lora_alpha": 16, "target_modules": ["q_proj", "v_proj"]}
        where = vibevoice.lora_dir(found["id"])
        assert where == paths.vibevoice_loras_root() / found["id"]
        assert sorted(child.name for child in where.iterdir()) == [
            "adapter_config.json", "adapter_model.safetensors", "meta.json"]
        assert (where / "adapter_model.safetensors").read_bytes() == \
            (source / "adapter_model.safetensors").read_bytes()
        meta = json.loads((where / "meta.json").read_text(encoding="utf-8"))
        assert (meta["name"], meta["parts"], meta["bytes"]) == ("my-voice-lora", ["llm"],
                                                                found["bytes"])
        assert meta["adapter"]["r"] == 8 and meta["created"] == found["created"]
        assert vibevoice.loras() == [found]
        assert _library() == [found["id"]], "no staging left behind"
        assert vibevoice.add_lora(str(source / "adapter_config.json"))["name"] == \
            "my-voice-lora (2)", "a file is taken as its folder; a derived name is made unique"

    def test_a_training_runs_layout_is_normalised_with_its_extras(self, voice_root, tmp_path):
        run = tmp_path / "run-7"
        _adapter(run / "lora")
        (run / "lora" / "README.md").write_text("not copied")
        (run / "lora" / "diffusion_head").mkdir()
        (run / "lora" / "diffusion_head" / "diffusion_head_full.bin").write_bytes(
            b"PK\x03\x04" + b"\0" * 64)
        (run / "lora" / "diffusion_head" / "config.json").write_text("{}")
        _adapter(run / "lora" / "semantic_connector")
        (run / "lora" / "acoustic_connector").mkdir()
        (run / "lora" / "acoustic_connector" / "pytorch_model.bin").write_bytes(
            b"\x80\x02" + b"\0" * 64)
        found = vibevoice.add_lora(str(run))
        assert found["name"] == "run-7"
        assert found["parts"] == ["llm", "diffusion_head", "acoustic_connector",
                                  "semantic_connector"]
        where = vibevoice.lora_dir(found["id"])
        assert (where / "adapter_config.json").is_file()
        assert (where / "adapter_model.safetensors").is_file()
        assert not (where / "lora").exists() and not (where / "README.md").exists()
        assert sorted(child.name for child in (where / "diffusion_head").iterdir()) == [
            "diffusion_head_full.bin"]
        assert sorted(child.name for child in (where / "acoustic_connector").iterdir()) == [
            "pytorch_model.bin"]
        assert sorted(child.name for child in (where / "semantic_connector").iterdir()) == [
            "adapter_config.json", "adapter_model.safetensors"]
        assert vibevoice.add_lora(str(run / "lora"))["name"] == "run-7 (2)", \
            "named for the run, not for its lora folder"

    def test_a_language_model_subfolder_with_extras_beside_it(self, voice_root, tmp_path):
        run = tmp_path / "finetune"
        _adapter(run / "language_model", weights="bin")
        (run / "diffusion_head").mkdir(parents=True)
        (run / "diffusion_head" / "model.safetensors").write_bytes(_safetensors())
        found = vibevoice.add_lora(str(run))
        assert found["parts"] == ["llm", "diffusion_head"]
        where = vibevoice.lora_dir(found["id"])
        assert (where / "adapter_model.bin").is_file()
        assert (where / "diffusion_head" / "model.safetensors").is_file()
        again = vibevoice.add_lora(str(run / "language_model"), name="pointed at the subfolder")
        assert again["parts"] == ["llm", "diffusion_head"], "the run's folder is one up"

    @pytest.mark.parametrize("layout, words", [
        ("empty", "holds no LoRA adapter"),
        ("no weights", "has an adapter_config.json but no adapter_model"),
        ("ia3", "is a PEFT IA3, and VibeVoice takes LoRA adapters only"),
        ("not json", "not readable JSON"),
        ("a list", "not an adapter configuration"),
        ("rank zero", "rank"),
        ("html weights", "not a readable weights file"),
        ("truncated weights", "not a readable weights file"),
        ("header not json", "not a readable weights file"),
        ("junk bin", "not a readable weights file"),
        ("empty extra", "holds none of"),
        ("extra adapter of another kind", "takes LoRA adapters only"),
        ("bad extra weights", "not a readable weights file"),
    ])
    def test_anything_else_is_refused_and_nothing_is_written(self, voice_root, tmp_path, layout,
                                                             words):
        folder = tmp_path / "candidate"
        folder.mkdir()
        if layout == "no weights":
            _adapter(folder, weights="")
        elif layout == "ia3":
            _adapter(folder, peft_type="IA3")
        elif layout == "not json":
            _adapter(folder)
            (folder / "adapter_config.json").write_text("{not json")
        elif layout == "a list":
            _adapter(folder)
            (folder / "adapter_config.json").write_text("[1, 2]")
        elif layout == "rank zero":
            _adapter(folder, rank=0)
        elif layout == "html weights":
            _adapter(folder)
            (folder / "adapter_model.safetensors").write_bytes(b"<!DOCTYPE html><html></html>")
        elif layout == "truncated weights":
            _adapter(folder)
            (folder / "adapter_model.safetensors").write_bytes(struct.pack("<Q", 100) + b"{}")
        elif layout == "header not json":
            _adapter(folder)
            (folder / "adapter_model.safetensors").write_bytes(
                struct.pack("<Q", 10) + b"not json!!" + b"\0" * 32)
        elif layout == "junk bin":
            _adapter(folder, weights="bin")
            (folder / "adapter_model.bin").write_bytes(b"version https://git-lfs.github" * 3)
        elif layout == "empty extra":
            _adapter(folder)
            (folder / "diffusion_head").mkdir()
            (folder / "diffusion_head" / "config.json").write_text("{}")
        elif layout == "extra adapter of another kind":
            _adapter(folder)
            _adapter(folder / "acoustic_connector", peft_type="PREFIX_TUNING")
        elif layout == "bad extra weights":
            _adapter(folder)
            (folder / "semantic_connector").mkdir()
            (folder / "semantic_connector" / "model.safetensors").write_bytes(b"\0" * 4)
        with pytest.raises(vibevoice.VibeVoiceError, match=re.escape(words)):
            vibevoice.add_lora(str(folder))
        assert vibevoice.loras() == [] and _library() == []

    def test_a_missing_folder_blank_input_and_too_large_are_refused(self, voice_root, tmp_path,
                                                                    monkeypatch):
        with pytest.raises(vibevoice.VibeVoiceError, match="Give the folder"):
            vibevoice.add_lora("  ")
        with pytest.raises(vibevoice.VibeVoiceError, match="not a folder"):
            vibevoice.add_lora(str(tmp_path / "nowhere"))
        source = _adapter(tmp_path / "big")
        monkeypatch.setattr(vibevoice, "LORA_BYTES_MAX", 64)
        with pytest.raises(vibevoice.VibeVoiceError, match="at most 64 B"):
            vibevoice.add_lora(str(source))
        assert _library() == []

    def test_names_are_one_line_bounded_and_unique(self, voice_root, tmp_path):
        source = _adapter(tmp_path / "base")
        first = vibevoice.add_lora(str(source), name="  Warm   narrator ")
        assert first["name"] == "Warm narrator"
        for name, words in [("warm NARRATOR", "already in the library"),
                            ("two\nlines", "one line"), ("x" * 81, "at most 80"),
                            ("   ", None)]:
            if words is None:
                assert vibevoice.add_lora(str(source), name=name)["name"] == "base"
                continue
            with pytest.raises(vibevoice.VibeVoiceError, match=words):
                vibevoice.add_lora(str(source), name=name)
        long_folder = _adapter(tmp_path / ("y" * 100))
        assert vibevoice.add_lora(str(long_folder))["name"] == "y" * 80

    def test_rename_keeps_names_unique_and_delete_forgets_the_adapter_everywhere(
            self, voice_root, tmp_path):
        one = vibevoice.add_lora(str(_adapter(tmp_path / "one")))
        two = vibevoice.add_lora(str(_adapter(tmp_path / "two")))
        assert vibevoice.rename_lora(one["id"], "Bright")["name"] == "Bright"
        assert vibevoice.rename_lora(one["id"], "BRIGHT")["name"] == "BRIGHT", "its own name"
        with pytest.raises(vibevoice.VibeVoiceError, match="already in the library"):
            vibevoice.rename_lora(two["id"], "bright")
        with pytest.raises(vibevoice.VibeVoiceError, match="needs a name"):
            vibevoice.rename_lora(two["id"], "")
        with pytest.raises(vibevoice.VibeVoiceError, match="not in the library"):
            vibevoice.rename_lora("0123456789ab", "x")
        vibevoice.set_settings({"lora_id": one["id"], "lora_scale": 0.4,
                                "chat": {"lora_id": two["id"], "lora_scale": 1.6}})
        gone = vibevoice.delete_lora(one["id"])
        assert gone["id"] == one["id"] and gone["name"] == "BRIGHT"
        assert [item["id"] for item in vibevoice.loras()] == [two["id"]]
        assert not (paths.vibevoice_loras_root() / one["id"]).exists()
        stored = json.loads(paths.vibevoice_settings_path().read_text(encoding="utf-8"))
        assert (stored["lora_id"], stored["lora_scale"]) == ("", 1.0), "cleared, not left"
        assert (stored["chat"]["lora_id"], stored["chat"]["lora_scale"]) == (two["id"], 1.6), \
            "a scope that named another LoRA keeps it"
        vibevoice.delete_lora(two["id"])
        stored = json.loads(paths.vibevoice_settings_path().read_text(encoding="utf-8"))
        assert (stored["chat"]["lora_id"], stored["chat"]["lora_scale"]) == ("", 1.0)
        found = vibevoice.settings()
        assert (found["lora_id"], found["lora_scale"]) == ("", 1.0)
        assert (found["chat"]["lora_id"], found["chat"]["lora_scale"]) == ("", 1.0)
        # What the page read back can be sent back: nothing in it names a LoRA that is gone.
        assert vibevoice.set_settings(vibevoice.settings()) == found
        with pytest.raises(vibevoice.VibeVoiceError, match="not in the library"):
            vibevoice.delete_lora(one["id"])
        with pytest.raises(vibevoice.VibeVoiceError, match="not in the library"):
            vibevoice.lora_dir("../loras")

    def test_the_library_lists_only_whole_adapters(self, voice_root, tmp_path):
        kept = vibevoice.add_lora(str(_adapter(tmp_path / "kept")))
        broken = vibevoice.add_lora(str(_adapter(tmp_path / "broken")))
        (vibevoice.lora_dir(broken["id"]) / "adapter_model.safetensors").unlink()
        root = paths.vibevoice_loras_root()
        adding = _adapter(root / ".adding-0123456789ab")
        (adding / "meta.json").write_text(json.dumps({"name": "half-added"}))
        (root / "notes.txt").write_text("a file")
        _adapter(root / "abcdef012345")  # an adapter with no record
        assert [item["id"] for item in vibevoice.loras()] == [kept["id"]]
        with pytest.raises(vibevoice.VibeVoiceError):
            vibevoice.lora_dir(broken["id"])

    def test_an_uninstall_keeps_the_library(self, windows, tmp_path):
        kept = vibevoice.add_lora(str(_adapter(tmp_path / "kept")))
        vibevoice.uninstall()
        assert vibevoice.loras() == [kept]


HUB_RT = "https://huggingface.co/microsoft/VibeVoice-Realtime-0.5B/resolve/main/"
HUB_QWEN = "https://huggingface.co/Qwen/Qwen2.5-0.5B/resolve/main/"


def _lend_realtime(monkeypatch, count=3) -> dict:
    """The Realtime model with its first ``count`` presets lent the digests of bytes a
    test can serve; returns ``{url: bytes}`` for those voices."""
    found = json.loads(json.dumps(vibevoice.manifest(refresh=True)))
    voices = found["models"][REALTIME]["voices"][:count]
    served = {}
    for item in voices:
        data = b"PK\x03\x04" + item["id"].encode("utf-8") * 8
        item["bytes"], item["sha256"] = len(data), _sha(data)
        served[item["url"]] = data
    found["models"][REALTIME]["voices"] = voices
    monkeypatch.setattr(vibevoice, "manifest", lambda refresh=False: found)
    return served


def _realtime_hub(index=False, upstream=True) -> dict:
    """What the Realtime model's repository and its tokenizer's would serve."""
    served = {HUB_RT + "config.json": json.dumps({"model_type": "vibevoice_streaming"}).encode()}
    if upstream:
        served[HUB_RT + "preprocessor_config.json"] = json.dumps({
            "processor_class": "VibeVoiceStreamingProcessor",
            "language_model_pretrained_name": "Qwen/Qwen2.5-0.5B"}).encode()
    if index:
        served[HUB_RT + "model.safetensors.index.json"] = json.dumps(_index()).encode()
        for name in SHARDS:
            served[HUB_RT + name] = _safetensors()
    else:
        served[HUB_RT + "model.safetensors"] = _safetensors()
    for name in TOKENIZER_FILES:
        served[HUB_QWEN + name] = _files()[name]
    return served


@pytest.fixture
def no_runtime_stop(monkeypatch):
    """Installing a model stops only the workers holding it, never the runtime whole."""

    def refuse(reason):
        raise AssertionError("a model install stopped every worker")

    monkeypatch.setattr(vibevoice, "_stop_runtime", refuse)
    monkeypatch.setattr(vibevoice, "_stop_workers_holding", lambda identifier, reason: None)


class TestInstallingTheRealtimeModel:
    def test_the_weights_file_the_tokenizer_and_the_presets_land_together(
            self, windows, net, local_pins, monkeypatch, no_runtime_stop):
        net.served.update(_realtime_hub())
        net.served.update(_lend_realtime(monkeypatch))
        said = []
        found = vibevoice.install("model", on_status=said.append, model_id=REALTIME)
        root = paths.vibevoice_model_root(REALTIME)
        assert (root / "model.safetensors").read_bytes() == _safetensors()
        assert not (root / "model.safetensors.index.json").exists()
        tokenizer = root / paths.VIBEVOICE_REALTIME_TOKENIZER_DIRNAME
        for name in TOKENIZER_FILES:
            assert (tokenizer / name).read_bytes() == net.served[HUB_QWEN + name]
        voices = vibevoice.bundle(REALTIME).voices
        for item in voices:
            assert (root / "voices" / item.filename).read_bytes() == net.served[item.artifact.url]
        local = json.loads((root / paths.VIBEVOICE_LOCAL_CONFIG).read_text(encoding="utf-8"))
        assert local["processor_class"] == "VibeVoiceStreamingProcessor"
        assert local["language_model_pretrained_name"] == str(tokenizer)
        assert "Qwen/Qwen2.5-0.5B" not in json.dumps(local), "the hub name would be fetched"
        assert (root / paths.VIBEVOICE_UPSTREAM_CONFIG).is_file()
        marker = json.loads((root / paths.INSTALLED_FILENAME).read_text(encoding="utf-8"))
        assert (marker["kind"], marker["weights"], marker["shards"]) == (
            "realtime", "single", ["model.safetensors"])
        assert marker["voices"] == [item.identifier for item in voices]
        assert marker["verified_by"][f"voices/{voices[0].filename}"] == \
            "this extension's manifest"
        assert found.model_installed and found.tokenizer_installed
        assert found.presets_installed == len(voices)
        assert not paths.vibevoice_model_root(MODEL).exists(), "the other model is untouched"
        assert any("weights are one file" in text for text in said)
        assert net.asked.index(HUB_RT + "model.safetensors.index.json") < \
            net.asked.index(HUB_RT + "model.safetensors"), "the index is asked for first"
        assert ("HEAD", voices[0].artifact.url) not in net.asked, \
            "a pinned voice is checked against this repository's digest, nobody's word"
        pins = json.loads(local_pins.read_text(encoding="utf-8"))["artifacts"]
        assert "microsoft/VibeVoice-Realtime-0.5B/model.safetensors" in pins
        assert "Qwen/Qwen2.5-0.5B/tokenizer.json" in pins
        assert not any(key.endswith(".pt") for key in pins), "the voices are pinned already"

    def test_an_index_when_one_is_published_names_the_weights(
            self, windows, net, local_pins, monkeypatch, no_runtime_stop):
        net.served.update(_realtime_hub(index=True))
        net.served.update(_lend_realtime(monkeypatch))
        vibevoice.install_model(model_id=REALTIME)
        root = paths.vibevoice_model_root(REALTIME)
        marker = json.loads((root / paths.INSTALLED_FILENAME).read_text(encoding="utf-8"))
        assert (marker["weights"], marker["shards"]) == ("index", list(SHARDS))
        assert HUB_RT + "model.safetensors" not in net.asked
        assert vibevoice.status(REALTIME).model_message.startswith(
            "Installed — VibeVoice Realtime 0.5B, 2 shards")

    def test_the_7b_still_needs_its_index(self, windows, net, local_pins, monkeypatch,
                                          no_runtime_stop):
        with pytest.raises(models.VoiceError):
            vibevoice.install_model(model_id=MODEL)
        assert not paths.vibevoice_model_root(MODEL).exists()
        entry = vibevoice.bundle(MODEL)
        assert entry.single is None and not entry.index_optional

    def test_a_preset_that_arrives_wrong_installs_nothing(self, windows, net, local_pins,
                                                          monkeypatch, no_runtime_stop):
        net.served.update(_realtime_hub())
        served = _lend_realtime(monkeypatch)
        url = list(served)[1]
        served[url] = served[url][:-3] + b"xyz"
        net.served.update(served)
        with pytest.raises(models.VoiceError, match="not what"):
            vibevoice.install_model(model_id=REALTIME)
        assert not paths.vibevoice_model_root(REALTIME).exists()
        assert not local_pins.exists(), "nothing that did not install is recorded"

    def test_a_preset_that_is_not_a_voice_prompt_is_refused_whatever_its_digest(
            self, windows, net, local_pins, monkeypatch, no_runtime_stop):
        found = json.loads(json.dumps(vibevoice.manifest(refresh=True)))
        voice = found["models"][REALTIME]["voices"][0]
        data = b"<html>not a voice</html>"
        voice["bytes"], voice["sha256"] = len(data), _sha(data)
        found["models"][REALTIME]["voices"] = [voice]
        monkeypatch.setattr(vibevoice, "manifest", lambda refresh=False: found)
        net.served.update(_realtime_hub())
        net.served[voice["url"]] = data
        with pytest.raises(vibevoice.VibeVoiceError, match="not a saved voice prompt"):
            vibevoice.install_model(model_id=REALTIME)
        assert not paths.vibevoice_model_root(REALTIME).exists()

    def test_missing_presets_are_put_back_alone(self, windows, net, local_pins, monkeypatch,
                                                no_runtime_stop):
        net.served.update(_realtime_hub())
        net.served.update(_lend_realtime(monkeypatch))
        vibevoice.install_model(model_id=REALTIME)
        root = paths.vibevoice_model_root(REALTIME)
        voice = vibevoice.bundle(REALTIME).voices[1]
        (root / "voices" / voice.filename).unlink()
        weights = (root / "model.safetensors").stat().st_mtime_ns
        net.asked.clear()
        said = []
        vibevoice.install_model(on_status=said.append, model_id=REALTIME)
        assert net.asked == [voice.artifact.url], "only the missing voice is fetched"
        assert (root / "voices" / voice.filename).read_bytes() == net.served[voice.artifact.url]
        assert (root / "model.safetensors").stat().st_mtime_ns == weights
        assert vibevoice.status(REALTIME).presets_installed == 3
        marker = json.loads((root / paths.INSTALLED_FILENAME).read_text(encoding="utf-8"))
        assert marker["digests"][f"voices/{voice.filename}"] == voice.artifact.sha256
        assert any("putting it back" in text for text in said)
        assert not any(child.name.startswith(f"{REALTIME}-voices")
                       for child in paths.vibevoice_staging_root().iterdir())
        net.asked.clear()
        said.clear()
        vibevoice.install_model(on_status=said.append, model_id=REALTIME)
        assert net.asked == [] and said == ["VibeVoice Realtime 0.5B is already installed."]

    def test_a_folder_with_the_presets_is_checked_against_their_pins(
            self, windows, tmp_path, net, local_pins, monkeypatch, no_runtime_stop):
        served = _lend_realtime(monkeypatch)
        folder = tmp_path / "rt"
        (folder / paths.VIBEVOICE_REALTIME_TOKENIZER_DIRNAME).mkdir(parents=True)
        (folder / "demo" / "voices" / "streaming_model").mkdir(parents=True)
        for url, data in _realtime_hub().items():
            name = url.rsplit("/", 1)[-1]
            where = folder / (f"{paths.VIBEVOICE_REALTIME_TOKENIZER_DIRNAME}/{name}"
                              if url.startswith(HUB_QWEN) else name)
            where.write_bytes(data)
        for url, data in served.items():
            (folder / "demo" / "voices" / "streaming_model" / url.rsplit("/", 1)[-1]) \
                .write_bytes(data)
        found = vibevoice.install_from("model", str(folder), model_id=REALTIME)
        assert net.asked == [], "a complete folder reaches for nothing"
        assert found.ready is False and found.model_installed and found.presets_installed == 3
        marker = json.loads((paths.vibevoice_model_root(REALTIME) / paths.INSTALLED_FILENAME)
                            .read_text(encoding="utf-8"))
        assert marker["source"] == "local" and marker["weights"] == "single"
        shutil_rmtree(paths.vibevoice_model_root(REALTIME))
        first = list(served)[0].rsplit("/", 1)[-1]
        (folder / "demo" / "voices" / "streaming_model" / first).write_bytes(b"PK\x03\x04 other")
        with pytest.raises(vibevoice.VibeVoiceError, match="not the preset voice this extension"):
            vibevoice.install_from("model", str(folder), model_id=REALTIME)
        assert not paths.vibevoice_model_root(REALTIME).exists()

    def test_a_folder_without_presets_fetches_them_or_says_where_to_put_them(
            self, windows, tmp_path, net, local_pins, monkeypatch, no_runtime_stop):
        served = _lend_realtime(monkeypatch)
        folder = tmp_path / "rt"
        (folder / "tokenizer").mkdir(parents=True)
        for url, data in _realtime_hub().items():
            name = url.rsplit("/", 1)[-1]
            (folder / ("tokenizer/" + name if url.startswith(HUB_QWEN) else name)).write_bytes(
                data)
        net.served.update(served)
        found = vibevoice.install_from("model", str(folder), model_id=REALTIME)
        assert found.presets_installed == 3
        assert sorted(net.asked) == sorted(served), "only the voices came from the network"
        shutil_rmtree(paths.vibevoice_model_root(REALTIME))
        net.served.clear()
        with pytest.raises(vibevoice.VibeVoiceError) as caught:
            vibevoice.install_from("model", str(folder), model_id=REALTIME)
        assert str(folder / "voices") in str(caught.value)
        assert f"at commit {COMMIT[:12]}" in str(caught.value)
        assert "demo/voices/streaming_model" in str(caught.value)

    def test_install_routes_the_model_part_to_the_model_it_names(self, windows, monkeypatch):
        seen = []
        monkeypatch.setattr(vibevoice, "install_model",
                            lambda on_status=None, on_progress=None, folder=None, model_id="":
                            seen.append((model_id, folder)))
        vibevoice.install("model", model_id=REALTIME)
        assert vibevoice.progress()["model"] == REALTIME
        vibevoice.install_from("model", "/somewhere", model_id=REALTIME)
        vibevoice.install("model")
        assert seen == [(REALTIME, None), (REALTIME, "/somewhere"), (MODEL, None)]
        with pytest.raises(vibevoice.VibeVoiceError, match="not a VibeVoice model"):
            vibevoice.install("model", model_id="vibevoice-1.5b")
        with pytest.raises(vibevoice.VibeVoiceError, match="not a VibeVoice model"):
            vibevoice.install_from("model", "/somewhere", model_id="vibevoice-1.5b")
        assert len(seen) == 3, "an unknown model starts nothing"
        assert vibevoice.progress()["running"] is False

    def test_the_manual_sources_of_the_realtime_model_include_its_presets(self, windows):
        found = vibevoice.sources("model", model_id=REALTIME)
        names = [row["save_as"] for row in found]
        assert "model.safetensors" in names and "config.json" in names
        assert f"{paths.VIBEVOICE_REALTIME_TOKENIZER_DIRNAME}/merges.txt" in names
        voices = [row for row in found if row["save_as"].startswith("voices/")]
        assert len(voices) == 25
        assert voices[0] == {"filename": "en-Carter_man.pt", "save_as": "voices/en-Carter_man.pt",
                             "url": RAW + "demo/voices/streaming_model/en-Carter_man.pt",
                             "archive": False}
        assert not any(row["save_as"].startswith("voices/")
                       for row in vibevoice.sources("model", model_id=MODEL))


class TestInstallingOneModelLeavesTheOthersWorker:
    def _runtime(self, monkeypatch, loaded=True):
        stopped = []
        held = {"GPU-aa": {"model_id": MODEL}, "GPU-bb": {"model_id": REALTIME}, "GPU-cc": None}
        fake = types.SimpleNamespace(
            status=lambda card_uuid="": {"cards": {key: {"uuid": key} for key in held}},
            stop=lambda card_uuid="", reason="": stopped.append(card_uuid))
        if loaded:
            fake.loaded = lambda card: held[card]
        monkeypatch.setitem(sys.modules, "mc_voice_vibevoice_runtime", fake)
        return stopped

    def test_only_the_worker_holding_the_model_is_stopped(self, monkeypatch):
        stopped = self._runtime(monkeypatch)
        vibevoice._stop_workers_holding(REALTIME, "the model is being installed")
        assert stopped == ["GPU-bb"]
        vibevoice._stop_workers_holding(MODEL, "the model is being installed")
        assert stopped == ["GPU-bb", "GPU-aa"]

    def test_a_runtime_that_cannot_say_what_it_holds_is_stopped_whole(self, monkeypatch):
        stopped = self._runtime(monkeypatch, loaded=False)
        vibevoice._stop_workers_holding(REALTIME, "the model is being installed")
        assert stopped == [""]

    def test_a_model_install_asks_about_its_own_model(self, windows, net, local_pins,
                                                       monkeypatch):
        asked = []
        monkeypatch.setattr(vibevoice, "_stop_workers_holding",
                            lambda identifier, reason: asked.append(identifier))
        monkeypatch.setattr(vibevoice, "_stop_runtime",
                            lambda reason: pytest.fail("every worker was stopped"))
        net.served.update(_realtime_hub())
        net.served.update(_lend_realtime(monkeypatch))
        vibevoice.install_model(model_id=REALTIME)
        assert asked == [REALTIME]


def _raw_served(tool) -> dict:
    """What raw.githubusercontent.com would serve, lent: the overlay and every preset."""
    served = {tool.raw_url(path): _overlay_bytes(path) for path, _why in tool.OVERLAY}
    for stem in tool.PRESETS:
        served[tool.raw_url(f"{tool.PRESETS_PATH}/{stem}.pt")] = b"PK\x03\x04" + stem.encode()
    return served


class TestThePinToolsOverlayAndPresets:
    def test_overlay_entries_are_sized_and_hashed_from_what_arrived(self):
        tool = _tool()
        served = _raw_served(tool)
        asked = []
        raw = tool.Raw(fetch=lambda url: asked.append(url) or served[url])
        found = tool.overlay_entries(raw, lambda text: None)
        assert [item["path"] for item in found] == list(OVERLAY_PATHS)
        for item in found:
            data = _overlay_bytes(item["path"])
            assert (item["bytes"], item["sha256"]) == (len(data), _sha(data))
            assert item["url"] == RAW + item["path"]
            assert item["source"] == {"repository": "https://github.com/microsoft/VibeVoice",
                                      "commit": COMMIT}
            assert item["why"]
        assert asked == [RAW + path for path in OVERLAY_PATHS]
        tool.overlay_entries(raw, lambda text: None)
        assert len(asked) == len(OVERLAY_PATHS), "read once, then cached"
        with pytest.raises(tool.PinError, match="not a Python source file"):
            tool.overlay_entries(tool.Raw(fetch=lambda url: b"\xff\xfe\x00"), lambda text: None)

    def test_preset_entries_carry_what_each_name_says(self):
        tool = _tool()
        served = _raw_served(tool)
        found = tool.preset_entries(tool.Raw(fetch=served.__getitem__), lambda text: None)
        assert [item["id"] for item in found] == list(tool.PRESETS)
        for item in found:
            data = served[item["url"]]
            assert (item["bytes"], item["sha256"]) == (len(data), _sha(data))
            assert item["url"] == f"{RAW}demo/voices/streaming_model/{item['id']}.pt"
        with pytest.raises(tool.PinError, match="not a torch.save archive"):
            tool.preset_entries(tool.Raw(fetch=lambda url: b"version https://git-lfs.github.com"),
                                lambda text: None)

    @pytest.mark.parametrize("stem, meta", [
        ("en-Carter_man", {"name": "Carter", "language": "en", "accent": "",
                           "gender": "man", "experimental": False}),
        ("in-Samuel_man", {"name": "Samuel", "language": "en", "accent": "in",
                           "gender": "man", "experimental": False}),
        ("de-Spk1_woman", {"name": "Speaker 1", "language": "de", "accent": "",
                           "gender": "woman", "experimental": True}),
        ("sp-Spk0_woman", {"name": "Speaker 0", "language": "sp", "accent": "",
                           "gender": "woman", "experimental": True}),
    ])
    def test_preset_meta(self, stem, meta):
        assert _tool().preset_meta(stem) == meta

    @pytest.mark.parametrize("stem", ["xx-Carter_man", "en-Carter", "en-Carter_child",
                                      "en-_man", "en-Car ter_man", "Carter"])
    def test_a_preset_name_it_cannot_read_is_refused(self, stem):
        tool = _tool()
        with pytest.raises(tool.PinError, match="not a preset voice name"):
            tool.preset_meta(stem)

    def _offline(self, tool, monkeypatch, existing):
        """The closure as written, re-read without the network, so ``build`` can run."""
        wheels = [item for item in existing["runtime"]["platforms"][0]["artifacts"]
                  if item.get("filename")]

        def resolve_triples(rows, pypi, say, python=tool.PYTHON):
            chosen = {}
            for name, version, kind in rows:
                item = next(one for one in wheels
                            if _normalise(one["filename"].split("-")[0]) == _normalise(name)
                            and one["filename"].split("-")[1] == version)
                chosen[_normalise(name)] = tool.Choice(
                    name, version, kind, {"filename": item["filename"], "url": item["url"],
                                          "size": item["bytes"],
                                          "digests": {"sha256": item["sha256"]}}, ())
            return chosen

        monkeypatch.setattr(tool, "resolve_triples", resolve_triples)
        monkeypatch.setattr(tool, "requirements_met", lambda *args, **kwargs: None)
        return tool.PyPI(project=lambda name: {},
                         release=lambda name, version: {"info": {"requires_dist": []}})

    def test_build_refuses_bytes_that_changed_under_a_committed_digest(self, monkeypatch):
        tool = _tool()
        existing = json.loads(json.dumps(vibevoice.manifest(refresh=True)))
        pypi = self._offline(tool, monkeypatch, existing)
        raw = tool.Raw(fetch=_raw_served(tool).__getitem__)
        _found, state = tool.build(existing, tool.Options(keep_closure=True, pypi=pypi, raw=raw),
                                   lambda text: None)
        assert len(state.disagreements) == len(OVERLAY_PATHS) + 25
        assert any("runtime.overlay/overlay:vibevoice/modular/configuration_vibevoice.py" in line
                   for line in state.disagreements)
        assert any(f"{REALTIME}/voice:en-Carter_man" in line for line in state.disagreements)

    def test_build_writes_the_overlay_and_the_presets_it_pinned(self, monkeypatch):
        tool = _tool()
        existing = json.loads(json.dumps(vibevoice.manifest(refresh=True)))
        del existing["runtime"]["overlay"]
        existing["models"][REALTIME]["voices"] = []
        pypi = self._offline(tool, monkeypatch, existing)
        served = _raw_served(tool)
        raw = tool.Raw(fetch=served.__getitem__)
        found, state = tool.build(existing, tool.Options(keep_closure=True, pypi=pypi, raw=raw),
                                  lambda text: None)
        assert state.disagreements == []
        assert state.overlay_pinned and state.presets_pinned and state.presets == 25
        assert state.models_hashed == {MODEL: False, REALTIME: False}
        assert state.pinned is False and found["pinned"] is False
        assert found["runtime"]["overlay"] == tool.overlay_entries(raw, lambda text: None)
        voices = found["models"][REALTIME]["voices"]
        assert [item["id"] for item in voices] == list(tool.PRESETS)
        assert voices[0]["sha256"] == _sha(b"PK\x03\x04en-Carter_man")
        assert found["models"][REALTIME]["voices_source"]["commit"] == COMMIT
        assert found["runtime"]["closure"] == existing["runtime"]["closure"]
        assert found["version"] == existing["version"] + 1
        assert "HASHED as well: the runtime overlay" in found["notes"]
        assert "25 preset voices" in found["notes"]

    def test_the_notes_say_when_the_overlay_or_the_presets_are_not_pinned(self):
        tool = _tool()
        state = tool.State(closure_complete=True)
        notes = tool.notes_for(state, {"version": "2.8.0"}, {})
        assert "NOT YET PINNED: the runtime overlay" in notes
        assert "NOT YET PINNED: the Realtime 0.5B's preset voices" in notes
        for field in ("overlay_pinned", "presets_pinned", "model_hashed", "torch_recorded"):
            done = tool.State(closure_complete=True, overlay_pinned=True, presets_pinned=True,
                              model_hashed=True, torch_recorded=True)
            setattr(done, field, False)
            assert done.pinned is False, field

    def test_model_takes_the_one_weights_file_when_no_index_is_published(self, monkeypatch):
        tool = _tool()
        entry = json.loads(json.dumps(vibevoice.manifest(refresh=True)))["models"][REALTIME]
        served = {"microsoft/VibeVoice-Realtime-0.5B": ["config.json", "model.safetensors",
                                                        "preprocessor_config.json"],
                  "Qwen/Qwen2.5-0.5B": list(TOKENIZER_FILES)}
        monkeypatch.setattr(tool, "repository", lambda repo, revision, say: (
            ("c" if "Realtime" in repo else "d") * 40, tuple(sorted(served[repo]))))

        def hub_artifact(repo, commit, path, local_name):
            return {"filename": path, "local_name": local_name,
                    "url": tool.HUB_RESOLVE.format(repo=repo, revision=commit, path=path),
                    "bytes": 1000 + len(path), "sha256": _sha(f"{repo}/{path}".encode())}

        monkeypatch.setattr(tool, "hub_artifact", hub_artifact)
        monkeypatch.setattr(tool, "_get", lambda url: pytest.fail("no index was published"))
        found = tool.model(entry, lambda text: None, REALTIME)
        assert [item["filename"] for item in found["shards"]] == ["model.safetensors"]
        assert found["single_weights"] == found["shards"][0]
        assert found["estimates"]["weights_bytes"] == 1000 + len("model.safetensors")
        assert found["estimates"]["weights_by_precision"]["bf16"] == 1000 + len(
            "model.safetensors")
        index = next(item for item in found["files"]
                     if item["filename"] == "model.safetensors.index.json")
        assert index.get("sha256") is None and index["optional"] is True
        assert tool._model_hashed(found)
        served["microsoft/VibeVoice-Realtime-0.5B"] = ["config.json"]
        with pytest.raises(tool.PinError,
                           match="serves no model.safetensors.index.json and no model.safetensors"):
            tool.model(entry, lambda text: None, REALTIME)

    def test_model_prefers_the_index_when_there_is_one(self, monkeypatch):
        tool = _tool()
        entry = json.loads(json.dumps(vibevoice.manifest(refresh=True)))["models"][REALTIME]
        served = {"microsoft/VibeVoice-Realtime-0.5B": ["config.json",
                                                        "model.safetensors.index.json"]
                  + list(SHARDS), "Qwen/Qwen2.5-0.5B": list(TOKENIZER_FILES)}
        monkeypatch.setattr(tool, "repository", lambda repo, revision, say: (
            "c" * 40, tuple(sorted(served[repo]))))
        monkeypatch.setattr(tool, "hub_artifact", lambda repo, commit, path, local_name: {
            "filename": path, "local_name": local_name, "url": f"https://huggingface.co/{repo}/"
            f"resolve/{commit}/{path}", "bytes": 7, "sha256": _sha(path.encode())})
        monkeypatch.setattr(tool, "_get", lambda url: (200, json.dumps(_index()).encode()))
        found = tool.model(entry, lambda text: None, REALTIME)
        assert [item["filename"] for item in found["shards"]] == list(SHARDS)
        assert found["single_weights"]["sha256"] is None, "the fallback stays declared"

    def test_two_models_files_of_one_name_are_two_artifacts(self):
        tool = _tool()
        existing = {"models": {
            "a": {"files": [{"filename": "config.json", "sha256": "a" * 64,
                             "url": "https://huggingface.co/o/a/resolve/main/config.json"}]},
            "b": {"files": [{"filename": "config.json", "sha256": "b" * 64,
                             "url": "https://huggingface.co/o/b/resolve/main/config.json"}]}}}
        assert tool._committed(existing) == {"o/a/config.json": "a" * 64,
                                             "o/b/config.json": "b" * 64}
        assert tool._identity({"filename": "config.json", "url":
                               "https://huggingface.co/o/a/resolve/" + "c" * 40 +
                               "/config.json"}) == "o/a/config.json", "whatever the revision"
        assert tool._identity({"path": "vibevoice/x.py", "url": RAW + "vibevoice/x.py"}) == \
            "overlay:vibevoice/x.py"
        assert tool._identity({"id": "en-Carter_man", "url": RAW + "demo/x.pt"}) == \
            "voice:en-Carter_man"
        clash = {"runtime": {"overlay": [{"path": "vibevoice/x.py", "sha256": "a" * 64},
                                         {"path": "vibevoice/x.py", "sha256": "b" * 64}]}}
        with pytest.raises(tool.PinError, match="two digests"):
            tool._committed(clash)
