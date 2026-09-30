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

import hashlib
import importlib.util
import json
import re
import shutil
import struct
import sys
from pathlib import Path

import pytest

import mc_voice_models as models
import mc_voice_paths as paths
import mc_voice_vibevoice as vibevoice
from vibevoice_worker import worker

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
        assert found == {"card_uuid": "", "model_id": MODEL, "steps": 12, "cfg_scale": 1.3,
                         "seed": None, "max_new_tokens": None, "keep_warm": True}

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
        assert vibevoice.settings()["steps"] == 12, "a refused change wrote nothing"

    def test_a_stored_value_that_no_longer_validates_falls_back_alone(self, voice_root):
        vibevoice._write_json(paths.vibevoice_settings_path(),
                              {"steps": 999, "cfg_scale": 2.0, "model_id": "gone"})
        found = vibevoice.settings()
        assert found["steps"] == 12
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
        found = vibevoice.calibration()[MODEL]
        assert found["renders"] == 3 and found["peak_bytes"] == 25_000_000_000
        assert found["rss_bytes"] == 4_000_000_000 and found["updated"]

    def test_a_batch_asks_for_the_weights_once_and_a_working_set_a_take(self, voice_root):
        weights, working = 18_700_000_000, 2 * 1024 ** 3
        assert vibevoice.need_vram_bytes(takes=4) == weights + 4 * working
        vibevoice.note_peak(MODEL, weights + 3_000_000_000)
        assert vibevoice.need_vram_bytes() == weights + 3_000_000_000
        assert vibevoice.need_vram_bytes(takes=4) == weights + 4 * 3_000_000_000

    def test_a_batchs_peak_is_kept_as_one_takes(self, voice_root):
        """Four takes reach the weights plus four working sets; what is kept is
        one take's, so a single render afterwards does not ask for four."""
        weights = 18_700_000_000
        vibevoice.note_peak(MODEL, weights + 4 * 3_000_000_000, takes=4)
        assert vibevoice.calibration()[MODEL]["peak_bytes"] == weights + 3_000_000_000
        assert vibevoice.need_vram_bytes(takes=4) == weights + 4 * 3_000_000_000

    def test_note_peak_writes_atomically_and_never_raises(self, voice_root):
        vibevoice.note_peak(MODEL, 1_000)
        path = paths.vibevoice_calibration_path()
        assert path.is_file()
        assert not path.with_name(path.name + ".new").exists()
        vibevoice.note_peak("", 5)
        vibevoice.note_peak("../escape", 5)
        vibevoice.note_peak(MODEL, "not a number")
        vibevoice.note_peak(MODEL, 0, 0)
        assert list(vibevoice.calibration()) == [MODEL]
        assert vibevoice.calibration()[MODEL]["renders"] == 1

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
# What a render may ask for
# --------------------------------------------------------------------------- #


class TestTheOptionsThePageLists:
    def test_every_solver_and_attention_the_worker_has_is_listed_with_upstreams_marked(
            self, voice_root):
        found = vibevoice.options()
        assert [entry["id"] for entry in found["solvers"]] == list(worker.SOLVERS)
        assert found["solvers"][0] == {"id": "dpmpp_2m", "name": "DPM++ 2M",
                                       "label": "DPM++ 2M (upstream)"}
        assert found["solvers"][1]["label"] == "DPM++ 2M SDE (upstream demo)"
        assert [entry["id"] for entry in found["attention"]] == list(worker.ATTENTION)
        assert found["batch_max"] == 4
        assert found["defaults"] == {"solver": "dpmpp_2m", "attention": "sdpa", "batch": 1,
                                     "steps": 12}
        assert vibevoice.public_status()["options"] == found

    def test_flash_attention_is_listed_unavailable_until_its_package_is_in_the_runtime(
            self, voice_root):
        def flash():
            return next(entry for entry in vibevoice.options()["attention"]
                        if entry["id"] == "flash_attention_2")

        assert flash() == {"id": "flash_attention_2", "name": "Flash attention 2",
                           "label": "Flash attention 2 (upstream)", "available": False,
                           "reason": "not installed"}
        assert all(entry["available"] for entry in vibevoice.options()["attention"]
                   if entry["id"] != "flash_attention_2")
        for site in (Path("Lib") / "site-packages",
                     Path("lib") / "python3.13" / "site-packages"):
            package = paths.vibevoice_runtime_root() / "env" / site / "flash_attn"
            package.mkdir(parents=True)
            (package / "__init__.py").write_text("", encoding="utf-8")
            assert flash()["available"] is True and flash()["reason"] == ""
            shutil.rmtree(paths.vibevoice_runtime_root())
            assert flash()["available"] is False


# --------------------------------------------------------------------------- #
# Refusals and the worker's environment
# --------------------------------------------------------------------------- #


class TestRefusals:
    def test_the_sentence_names_what_is_missing(self, windows):
        assert vibevoice.refusal() == "VibeVoice's runtime is not installed — install it below."
        _install_runtime_files()
        assert vibevoice.refusal() == "The VibeVoice 7B model is not installed — install it below."
        root = _install_model_files(tokenizer=False)
        assert vibevoice.refusal() == "VibeVoice's tokenizer is not installed — install the model again."
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
        assert found[-1]["filename"] == "torch-2.8.0+cu128-cp313-cp313-win_amd64.whl"
        assert found[-1]["url"] == "https://download.pytorch.org/whl/cu128/torch/"
        assert len(found) == len(vibevoice.platform().artifacts) + 1


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
    def doubles(self, windows, monkeypatch):
        """Every step that needs the network, a venv or a real interpreter, replaced."""
        state = {"fetched": [], "built": None, "smoked": None}
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

        monkeypatch.setattr(vibevoice, "_build_environment", build)

        def smoke(staging):
            state["smoked"] = staging
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
                            lambda on_status=None, on_progress=None, folder=None:
                            order.append("model"))
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
        vibevoice.install_runtime(folder=str(folder))
        record = json.loads(paths.vibevoice_runtime_manifest().read_text(encoding="utf-8"))
        assert record["torch_wheel"]["sha256"] == _sha(b"torch wheel")
        assert doubles["fetched"] == [], "a folder install downloads nothing"


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
        assert vibevoice.calibration()[MODEL]["peak_bytes"] == 123_456_789


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
                          shards=10)
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
