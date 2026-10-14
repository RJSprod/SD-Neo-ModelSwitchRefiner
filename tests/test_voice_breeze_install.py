"""Breeze TTS 2's installer, manifest and settings: the trust root and what it claims.

The manifest is checked for the shape of its claim: every PyPI wheel of the
closure named, sized and hashed for the one platform; torch and torchaudio 2.9.1
as versioned resolves from the publisher's CUDA 12.8 index; the community Flash
attention 2 wheel for exactly that torch and CPython 3.13 as a declared
download; Breeze's own code pinned file by file at one commit; and the model
declared by repository and revision, not hashed, with its licence said.

The installer is exercised end to end against a small manifest of fake
packages: a real virtual environment is built from fake wheels, Breeze's fake
code is put beside it from the "raw host", and the real worker's ``--selftest``
runs in it -- so the venv layout, the source folder, the ``COMMIT`` file, the
environment variable that joins them and Flash attention's import are all
proved, not mocked. The model is listed from a fake hub, file by file at a
commit, each file checked against the digest the hub states, and a second
install lists the same commit.

Nothing here imports torch, transformers, qwen_tts or Breeze's code, and nothing
reaches the network: the one door out (``mc_voice_models``) is replaced.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import struct
import sys
import zipfile
from pathlib import Path

import pytest

import mc_voice_breeze as breeze
import mc_voice_models as models
import mc_voice_paths as paths
from breeze_worker import worker

ROOT = Path(__file__).resolve().parent.parent
TOOL = ROOT / "tools" / "pin_breeze_models.py"
MODEL = "breeze-tts-2"
COMMIT = "6eba901d1845353802652c708a4740fdfc5baa75"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _git_oid(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def _tool():
    found = sys.modules.get("pin_breeze_models")
    if found is not None:
        return found
    spec = importlib.util.spec_from_file_location("pin_breeze_models", TOOL)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(autouse=True)
def fresh(tmp_path, monkeypatch):
    """No test reads or writes the checkout's overlay: it is the user's record."""
    monkeypatch.setattr(paths, "breeze_local_pins_path",
                        lambda: tmp_path / "managed-breeze-models.local.json")
    breeze._manifest_cache_clear()
    yield
    breeze._manifest_cache_clear()


@pytest.fixture
def windows(voice_root, monkeypatch):
    """The release target: 64-bit Windows, CPython 3.13, an NVIDIA driver present."""
    monkeypatch.setattr(models, "current_platform", lambda: ("windows", "amd64", "3.13"))
    monkeypatch.setattr(breeze, "_nvidia_present", lambda: True)
    return voice_root


# --------------------------------------------------------------------------- #
# The manifest
# --------------------------------------------------------------------------- #


class TestTheManifest:
    def test_one_platform_the_closure_hashed_and_three_wheels_not(self):
        found = breeze.manifest(refresh=True)
        assert found["schema"] == 1 and found["defaults"] == {"model": MODEL}
        platform, = found["runtime"]["platforms"]
        assert (platform["id"], platform["system"], platform["python"],
                platform["accelerator"]) == ("windows-x86_64-cp313-cu128", "windows", "3.13",
                                             "cuda")
        wheels = [item for item in platform["artifacts"]
                  if not item.get("resolve") and not item.get("declared")]
        assert len(wheels) == len(found["runtime"]["closure"]) >= 40
        for item in wheels:
            assert len(item["sha256"]) == 64 and item["bytes"] > 0, item["filename"]
            assert item["url"].startswith("https://files.pythonhosted.org/"), item["filename"]
            name = item["filename"]
            assert name.endswith(".whl") and ("none-any" in name or "win_amd64" in name
                                              or "abi3" in name), name
        resolves = {item["resolve"]["package"]: item["resolve"]
                    for item in platform["artifacts"] if item.get("resolve")}
        assert {name: (entry["index"], entry["version"]) for name, entry in resolves.items()} \
            == {"torch": ("https://download.pytorch.org/whl/cu128/torch/", "2.9.1"),
                "torchaudio": ("https://download.pytorch.org/whl/cu128/torchaudio/", "2.9.1")}
        flash, = [item for item in platform["artifacts"] if item.get("declared")]
        assert flash["filename"] == ("flash_attn-2.8.3+cu128torch2.9.1cxx11abiTRUE-cp313-"
                                     "cp313-win_amd64.whl")
        assert flash["url"].startswith("https://huggingface.co/Wildminder/AI-windows-whl/")

    def test_the_closure_is_what_qwen_tts_imports_and_not_its_demo(self):
        """qwen_tts imports librosa, sox, onnxruntime and einops when Breeze's
        loader imports its codec, so they ship; its demo's gradio does not."""
        found = breeze.manifest(refresh=True)
        names = {row[0].casefold() for row in found["runtime"]["closure"]}
        for needed in ("qwen-tts", "transformers", "accelerate", "librosa", "sox",
                       "onnxruntime", "einops", "soundfile", "numpy", "huggingface-hub"):
            assert needed in names, needed
        versions = {row[0].casefold(): row[1] for row in found["runtime"]["closure"]}
        assert (versions["qwen-tts"], versions["transformers"], versions["accelerate"]) == \
            ("0.1.1", "4.57.3", "1.12.0")
        for absent in ("gradio", "torch", "torchaudio", "hf-xet", "fastapi", "uvicorn"):
            assert absent not in names, absent
        assert set(found["runtime"]["excluded"]) == {"gradio", "torchaudio", "hf-xet"}
        assert all(found["runtime"]["excluded"].values())

    def test_breezes_code_is_pinned_file_by_file_at_the_commit(self):
        code = breeze.source()
        assert code["repo"] == "breezeblue-ai/breeze-tts" and code["commit"] == COMMIT
        paths_ = {row["path"] for row in code["files"]}
        for wanted in ("breeze_infer/__init__.py", "breeze_infer/runtime.py",
                       "breeze_infer/templates.py", "breeze_infer/audio.py",
                       "models/fast_streaming.py", "models/breeze.py", "configs/fast.json",
                       "LICENSE"):
            assert wanted in paths_, wanted
        assert "breeze_infer/api.py" not in paths_ and "infer.py" not in paths_
        for row in code["files"]:
            assert len(row["sha256"]) == 64
            assert row["url"] == f"https://raw.githubusercontent.com/breezeblue-ai/breeze-tts/" \
                                 f"{COMMIT}/{row['path']}"
        empty, = [row for row in code["files"] if row["path"] == "models/__init__.py"]
        assert empty["bytes"] == 0 and empty["sha256"] == _sha(b"")

    def test_the_model_is_declared_not_hashed_and_its_licence_said(self):
        entry = breeze.bundle(MODEL)
        assert (entry.repo, entry.revision, entry.authorized, entry.files) == \
            ("BreezeBlue/Breeze-TTS-2", "main", True, ())
        assert not entry.pinned
        assert "Non-Commercial" in entry.license and "Non-Commercial" in entry.attribution
        notes = breeze.manifest()["notes"]
        assert notes.startswith("PARTIALLY PINNED.")
        assert "DECLARED BUT NOT HASHED" in notes and "torch 2.9.1 and torchaudio 2.9.1" in notes

    def test_the_pin_tool_and_the_worker_agree_on_the_commit_and_the_wheel(self):
        tool = _tool()
        assert tool.COMMIT == breeze.source()["commit"] == COMMIT
        assert tool.FLASH_ATTN["filename"] == breeze.flash_artifact().filename \
            if breeze.flash_artifact() else True
        assert set(tool.SOURCE_FILES) <= {row["path"] for row in breeze.source()["files"]}
        assert tool.MANIFEST == paths.breeze_manifest_path()
        assert "managed-vibevoice-models.json" in tool.SIBLINGS

    def test_the_pin_tool_refuses_another_engines_manifest(self, tmp_path, capsys):
        tool = _tool()
        sibling = tmp_path / "managed-vibevoice-models.json"
        sibling.write_text("{}", encoding="utf-8")
        assert tool.main(["--check", "--manifest", str(sibling)]) == 2
        assert "belongs to another engine" in capsys.readouterr().err


# --------------------------------------------------------------------------- #
# A small world of fake packages, served by a fake network
# --------------------------------------------------------------------------- #


def _wheel(folder: Path, filename: str, distribution: str, version: str, files: dict) -> bytes:
    path = folder / filename
    with zipfile.ZipFile(path, "w") as bundle:
        for name, text in files.items():
            bundle.writestr(name, text)
        info = f"{distribution.replace('-', '_')}-{version}.dist-info"
        bundle.writestr(f"{info}/METADATA", f"Metadata-Version: 2.1\nName: {distribution}\n"
                                            f"Version: {version}\n")
        bundle.writestr(f"{info}/RECORD", "")
    return path.read_bytes()


SOURCE = {
    "breeze_infer/__init__.py": b"",
    "breeze_infer/templates.py": b"NAME = 'templates'\n",
    "breeze_infer/runtime.py": (b"def load_runtime(*args, **kwargs):\n    return None\n"
                                b"def set_all_seeds(seed):\n    return None\n"
                                b"def update_generation_config_for_breeze(model):\n"
                                b"    return None\n"),
    "models/__init__.py": b"",
    "models/fast_streaming.py": (b"class FastBreezeStreamingRuntime:\n    pass\n"
                                 b"class FastStreamingConfig:\n    pass\n"),
}


class World:
    """What the publishers serve, by URL, and what was asked for."""

    def __init__(self, folder: Path, flash_loads: bool = True):
        self.folder = folder
        folder.mkdir(parents=True, exist_ok=True)
        self.served: dict = {}
        self.asked: list = []
        self.wheels = {}
        flash = ("" if flash_loads else
                 "raise ImportError('DLL load failed while importing flash_attn_2_cuda')\n")
        for filename, distribution, version, files in (
                ("transformers-4.57.3-py3-none-any.whl", "transformers", "4.57.3",
                 {"transformers/__init__.py": "__version__ = '4.57.3'\n"}),
                ("qwen_tts-0.1.1-py3-none-any.whl", "qwen-tts", "0.1.1",
                 {"qwen_tts/__init__.py": ""}),
                ("torch-2.9.1+cu128-cp313-cp313-win_amd64.whl", "torch", "2.9.1+cu128",
                 {"torch/__init__.py": "__version__ = '2.9.1+cu128'\n"
                                       "class cuda:\n"
                                       "    @staticmethod\n"
                                       "    def is_available():\n"
                                       "        return False\n"}),
                ("torchaudio-2.9.1+cu128-cp313-cp313-win_amd64.whl", "torchaudio",
                 "2.9.1+cu128", {"torchaudio/__init__.py": ""}),
                ("flash_attn-2.8.3+cu128torch2.9.1cxx11abiTRUE-cp313-cp313-win_amd64.whl",
                 "flash-attn", "2.8.3", {"flash_attn/__init__.py": flash})):
            self.wheels[filename] = _wheel(folder, filename, distribution, version, files)
        self.pypi = {name: f"https://files.pythonhosted.org/packages/aa/{name}"
                     for name in ("transformers-4.57.3-py3-none-any.whl",
                                  "qwen_tts-0.1.1-py3-none-any.whl")}
        for name, url in self.pypi.items():
            self.served[url] = self.wheels[name]
        self.flash_name = ("flash_attn-2.8.3+cu128torch2.9.1cxx11abiTRUE-cp313-cp313-"
                           "win_amd64.whl")
        self.flash_url = ("https://huggingface.co/Wildminder/AI-windows-whl/resolve/main/"
                          "flash_attn-2.8.3%2Bcu128torch2.9.1cxx11abiTRUE-cp313-cp313-"
                          "win_amd64.whl")
        self.served[self.flash_url] = self.wheels[self.flash_name]
        for package in ("torch", "torchaudio"):
            name = f"{package}-2.9.1+cu128-cp313-cp313-win_amd64.whl"
            self.served[f"https://download.pytorch.org/whl/cu128/"
                        f"{package}-2.9.1%2Bcu128-cp313-cp313-win_amd64.whl"] = self.wheels[name]
        self.source = {}
        for path, data in SOURCE.items():
            url = f"https://raw.githubusercontent.com/breezeblue-ai/breeze-tts/{COMMIT}/{path}"
            self.source[path] = {"path": path, "url": url, "bytes": len(data),
                                 "sha256": _sha(data)}
            self.served[url] = data

    def page(self, url: str) -> str:
        package = url.rstrip("/").rsplit("/", 1)[-1]
        name = f"{package}-2.9.1+cu128-cp313-cp313-win_amd64.whl"
        quoted = name.replace("+", "%2B")
        return (f'<html><body><a href="/whl/cu128/{quoted}#sha256={_sha(self.wheels[name])}">'
                f'x</a><a href="/whl/cu128/{package}-2.9.1%2Bcu128-cp312-cp312-win_amd64.whl'
                f'#sha256={"1" * 64}">y</a></body></html>')

    def manifest(self) -> dict:
        found = json.loads(json.dumps(breeze.manifest(refresh=True)))
        platform = found["runtime"]["platforms"][0]
        artifacts = []
        for name, url in self.pypi.items():
            data = self.wheels[name]
            artifacts.append({"filename": name, "local_name": name, "url": url,
                              "bytes": len(data), "sha256": _sha(data)})
        artifacts += [item for item in platform["artifacts"] if item.get("resolve")]
        artifacts += [item for item in platform["artifacts"] if item.get("declared")]
        platform["artifacts"] = artifacts
        found["runtime"]["source"] = {"repo": "breezeblue-ai/breeze-tts", "commit": COMMIT,
                                      "license": "Apache-2.0",
                                      "files": list(self.source.values())}
        return found

    def download(self, artifact, destination, report, expected):
        """The shared downloader's contract: write it, check it, say its digest."""
        self.asked.append(artifact.url)
        if artifact.url not in self.served:
            raise models.VoiceError(f"{artifact.filename} could not be downloaded (HTTP 404).")
        data = self.served[artifact.url]
        if expected.size and len(data) != expected.size:
            raise models.VoiceError(f"{artifact.filename} arrived as {len(data)} bytes.")
        if expected.sha256 and _sha(data) != expected.sha256:
            raise models.VoiceError(f"{artifact.filename} downloaded, but its contents are "
                                    f"not what {expected.source} says they should be.")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
        report(len(data))
        return _sha(data)

    def resolve(self, artifact):
        if artifact.sha256 and artifact.size:
            return models.Expected(artifact.size, artifact.sha256, "this extension's manifest")
        data = self.served[artifact.url]
        return models.Expected(len(data), _sha(data), "the publisher, over HTTPS")


@pytest.fixture
def world(windows, tmp_path, monkeypatch):
    found = World(tmp_path / "served")
    manifest_path = tmp_path / "managed-breeze-models.json"
    manifest_path.write_text(json.dumps(found.manifest()), encoding="utf-8")
    monkeypatch.setattr(paths, "breeze_manifest_path", lambda: manifest_path)
    breeze._manifest_cache_clear()
    monkeypatch.setattr(models, "read_index_page", found.page)
    monkeypatch.setattr(models, "_download", found.download)
    monkeypatch.setattr(models, "_resolve", found.resolve)
    monkeypatch.setattr(models, "_declared_size", lambda url: len(found.served[url]))
    return found


class TestInstallingTheRuntime:
    def test_a_real_venv_is_built_and_the_real_worker_proves_it_imports_breeze(self, world):
        said = []
        breeze.install_runtime(on_status=said.append)
        record = json.loads(paths.breeze_runtime_manifest().read_text(encoding="utf-8"))
        assert record["closure"] == breeze.closure_id()
        assert record["breeze_commit"] == COMMIT
        assert (record["torch_version"], record["torchaudio_version"],
                record["transformers_version"], record["qwen_tts_version"]) == \
            ("2.9.1+cu128", "2.9.1+cu128", "4.57.3", "0.1.1")
        assert record["flash_attn"]["loads"] is True and record["flash_attn"]["version"] == \
            "2.8.3"
        assert record["flash_attn"]["sha256"] == _sha(world.wheels[world.flash_name])
        assert set(record["resolved"]) == {"torch-2.9.1+cu128-cp313-cp313-win_amd64.whl",
                                           "torchaudio-2.9.1+cu128-cp313-cp313-win_amd64.whl"}
        code = paths.breeze_source_root()
        assert (code / "COMMIT").read_text(encoding="utf-8").strip() == COMMIT
        for path, data in SOURCE.items():
            assert (code / path).read_bytes() == data, path
        assert not any(url.endswith("__init__.py") for url in world.asked), \
            "an empty file is written, never fetched"
        assert breeze.runtime_python() is not None
        assert breeze.flash_attention_installed() is True
        assert not (paths.breeze_runtime_root() / "wheels").exists()
        found = breeze.status()
        assert found.runtime_installed and not found.stale
        assert "Flash attention 2.8.3" in found.runtime_message
        pins = json.loads(paths.breeze_local_pins_path().read_text(encoding="utf-8"))
        assert pins["artifacts"][f"Wildminder/AI-windows-whl/{world.flash_name}"] == {
            "sha256": _sha(world.wheels[world.flash_name]),
            "bytes": len(world.wheels[world.flash_name])}
        assert said[-1] == "The Breeze runtime is installed."

    def test_a_flash_wheel_that_will_not_load_is_installed_and_said_not_offered(
            self, windows, tmp_path, monkeypatch):
        found = World(tmp_path / "served", flash_loads=False)
        manifest_path = tmp_path / "managed-breeze-models.json"
        manifest_path.write_text(json.dumps(found.manifest()), encoding="utf-8")
        monkeypatch.setattr(paths, "breeze_manifest_path", lambda: manifest_path)
        breeze._manifest_cache_clear()
        monkeypatch.setattr(models, "read_index_page", found.page)
        monkeypatch.setattr(models, "_download", found.download)
        monkeypatch.setattr(models, "_resolve", found.resolve)
        monkeypatch.setattr(models, "_declared_size", lambda url: len(found.served[url]))
        breeze.install_runtime()
        record = json.loads(paths.breeze_runtime_manifest().read_text(encoding="utf-8"))
        assert record["flash_attn"]["loads"] is False
        assert "DLL load failed" in record["flash_attn"]["error"]
        assert breeze.status().runtime_installed, "SDPA and Eager still render"
        assert breeze.flash_attention_installed() is False
        flash, = [option for option in breeze.fields()[1]["options"]
                  if option["id"] == "flash_attention_2"]
        assert (flash["available"], flash["reason"]) == (False,
                                                         "installed, but it would not load")
        assert "would not load" in breeze.refuse({"attention": "flash_attention_2"})
        assert breeze.refuse({"attention": "sdpa"}) == ""

    def test_a_source_file_that_is_not_the_pinned_one_installs_nothing(self, world):
        url = world.source["breeze_infer/runtime.py"]["url"]
        world.served[url] = b"#" * len(world.served[url])  # the same size, other bytes
        with pytest.raises(models.VoiceError, match="not what"):
            breeze.install_runtime()
        assert not paths.breeze_runtime_root().exists()
        assert not any(paths.breeze_staging_root().iterdir()) \
            if paths.breeze_staging_root().exists() else True

    def test_a_torch_the_index_does_not_list_installs_nothing(self, world, monkeypatch):
        monkeypatch.setattr(models, "read_index_page", lambda url: "<html><body></body></html>")
        with pytest.raises(breeze.BreezeError, match="lists no wheels"):
            breeze.install_runtime()
        assert breeze.runtime_python() is None

    def test_an_installed_runtime_from_another_build_is_stale_and_rebuilt(self, world):
        breeze.install_runtime()
        record = json.loads(paths.breeze_runtime_manifest().read_text(encoding="utf-8"))
        record["closure"] = "0" * 16
        breeze._write_json(paths.breeze_runtime_manifest(), record)
        found = breeze.status()
        assert found.stale and not found.runtime_installed
        assert found.message == breeze.STALE_MESSAGE == found.stale_message
        assert breeze.refusal() == breeze.STALE_MESSAGE
        assert breeze.public_status()["stale"] is True
        breeze.install_runtime()
        assert breeze.status().runtime_installed and not breeze.status().stale

    def test_a_folder_holds_the_wheels_and_a_clone_of_breeze(self, world, tmp_path):
        folder = tmp_path / "downloads"
        for name, data in world.wheels.items():
            (folder).mkdir(exist_ok=True)
            (folder / name).write_bytes(data)
        clone = folder / f"breeze-tts-{COMMIT}"
        for path, data in SOURCE.items():
            (clone / path).parent.mkdir(parents=True, exist_ok=True)
            (clone / path).write_bytes(data)
        world.asked.clear()
        found = breeze.install_from("runtime", str(folder))
        assert found.runtime_installed
        assert world.asked == [], "nothing was fetched: the folder had everything"

    def test_a_folder_whose_breeze_was_edited_is_refused(self, world, tmp_path):
        folder = tmp_path / "downloads"
        folder.mkdir()
        for name, data in world.wheels.items():
            (folder / name).write_bytes(data)
        for path, data in SOURCE.items():
            (folder / path).parent.mkdir(parents=True, exist_ok=True)
            (folder / path).write_bytes(data)
        (folder / "breeze_infer" / "templates.py").write_bytes(b"NAME = 'edited'\n")
        with pytest.raises(breeze.BreezeError, match="not the file this build pins"):
            breeze.install_from("runtime", str(folder))
        assert breeze.runtime_python() is None


# --------------------------------------------------------------------------- #
# The model, listed from the hub
# --------------------------------------------------------------------------- #


def _safetensors() -> bytes:
    header = json.dumps({"__metadata__": {"format": "pt"}}).encode("utf-8")
    return struct.pack("<Q", len(header)) + header + b"\x00" * 16


MODEL_FILES = {
    "config.json": json.dumps({"model_type": "breeze"}).encode("utf-8"),
    "tokenizer.json": json.dumps({"version": "1.0"}).encode("utf-8"),
    "model-00001-of-00002.safetensors": _safetensors(),
    "model-00002-of-00002.safetensors": _safetensors() + b"\x01",
    "audio_tokenizer/config.json": json.dumps({"model_type": "qwen3_tts_tokenizer"}).encode(),
    "audio_tokenizer/model.safetensors": _safetensors() + b"\x02",
    "README.md": b"# Breeze",
    "assets/demo.wav": b"RIFF",
}
LFS = {"model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors",
       "audio_tokenizer/model.safetensors"}
HUB_COMMIT = "c" * 40


class Hub:
    """The hub's revision and tree APIs, paginated, and its files at a commit."""

    def __init__(self, files=None):
        self.files = dict(MODEL_FILES if files is None else files)
        self.listed = 0
        self.downloads: list = []

    def tree(self) -> list:
        rows = [{"type": "directory", "path": "audio_tokenizer"}]
        for path, data in self.files.items():
            row = {"type": "file", "path": path, "size": len(data), "oid": _git_oid(data)}
            if path in LFS:
                row["lfs"] = {"oid": _sha(data), "size": len(data)}
            rows.append(row)
        return rows

    def read_json(self, url, ceiling=0, authorized=False, name=""):
        assert authorized is True, "the model is behind its publisher's terms"
        if "/revision/" in url:
            return {"sha": HUB_COMMIT}, ""
        assert f"/tree/{HUB_COMMIT}" in url, "the tree is read at the commit, not at main"
        self.listed += 1
        rows = self.tree()
        if "cursor=2" in url:
            return rows[4:], ""
        return rows[:4], url + "&cursor=2"

    def download(self, artifact, destination, report, expected):
        self.downloads.append(artifact.url)
        assert artifact.authorized is True
        assert f"/resolve/{HUB_COMMIT}/" in artifact.url
        path = artifact.url.split(f"/resolve/{HUB_COMMIT}/", 1)[1].replace("%2F", "/")
        data = self.files[path]
        if expected.size and len(data) != expected.size:
            raise models.VoiceError(f"{path} arrived as {len(data)} bytes.")
        if expected.sha256 and _sha(data) != expected.sha256:
            raise models.VoiceError(f"{path} is not what {expected.source} says.")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
        return _sha(data)


@pytest.fixture
def hub(windows, monkeypatch):
    found = Hub()
    monkeypatch.setattr(models, "read_json", found.read_json)
    monkeypatch.setattr(models, "_download", found.download)
    return found


class TestInstallingTheModel:
    def test_every_listed_file_is_fetched_at_the_commit_and_checked(self, hub):
        breeze.install_model()
        root = paths.breeze_model_root(MODEL)
        for path, data in MODEL_FILES.items():
            if path in ("README.md", "assets/demo.wav"):
                assert not (root / path).exists(), "the card and its media are not the model"
            else:
                assert (root / path).read_bytes() == data, path
        marker = json.loads((root / paths.INSTALLED_FILENAME).read_text(encoding="utf-8"))
        assert (marker["commit"], marker["repo"], marker["source"]) == \
            (HUB_COMMIT, "BreezeBlue/Breeze-TTS-2", "hub")
        assert marker["verified_by"]["config.json"] == "the hub's file list, over HTTPS"
        assert hub.listed == 2, "both pages of the tree were read"
        found = breeze.status()
        assert found.model_installed and HUB_COMMIT[:7] in found.model_message
        assert "non-commercial" in found.model_message
        # The listing is recorded, and the next install of this model on this
        # machine is the same commit, listed from the record and not the hub.
        record = json.loads(paths.breeze_local_pins_path().read_text(encoding="utf-8"))
        assert record["models"][MODEL]["commit"] == HUB_COMMIT
        assert {row["path"] for row in record["models"][MODEL]["files"]} == \
            set(MODEL_FILES) - {"README.md", "assets/demo.wav"}
        breeze.uninstall()
        hub.listed = 0
        breeze.install_model()
        assert hub.listed == 0 and breeze.status().model_installed

    def test_the_weights_the_turn_asks_for_are_the_installed_files(self, hub):
        before = breeze.need_vram_bytes(MODEL)
        breeze.install_model()
        weights = sum(len(data) for path, data in MODEL_FILES.items()
                      if path.endswith(".safetensors"))
        assert breeze.need_vram_bytes(MODEL) == weights + breeze.WORKING_BYTES_DEFAULT
        assert before == breeze.WEIGHTS_BYTES_DEFAULT + breeze.WORKING_BYTES_DEFAULT

    def test_a_small_file_that_is_not_the_blob_the_hub_names_installs_nothing(self, hub):
        listed = hub.tree

        def tampered():
            rows = listed()
            for row in rows:
                if row.get("path") == "config.json":
                    row["oid"] = "0" * 40
            return rows

        hub.tree = tampered
        with pytest.raises(breeze.BreezeError, match="not the file"):
            breeze.install_model()
        assert not paths.breeze_model_root(MODEL).exists()

    def test_a_weight_that_is_not_the_lfs_object_installs_nothing(self, hub):
        hub.files["model-00002-of-00002.safetensors"] = _safetensors() + b"\x09"
        listed = hub.tree

        def stale():
            rows = listed()
            for row in rows:
                if row.get("path") == "model-00002-of-00002.safetensors":
                    row["lfs"]["oid"] = "f" * 64
            return rows

        hub.tree = stale
        with pytest.raises(models.VoiceError):
            breeze.install_model()
        assert not paths.breeze_model_root(MODEL).exists()

    def test_a_listing_without_the_codec_is_refused_before_a_byte(self, hub):
        del hub.files["audio_tokenizer/config.json"], hub.files["audio_tokenizer/model.safetensors"]
        with pytest.raises(breeze.BreezeError, match="audio_tokenizer"):
            breeze.install_model()
        assert hub.downloads == []

    def test_a_gated_model_says_what_to_do(self, windows, monkeypatch):
        def gated(url, ceiling=0, authorized=False, name=""):
            raise models.Gated(name, 401)

        monkeypatch.setattr(models, "read_json", gated)
        with pytest.raises(models.Gated, match="Access token"):
            breeze.install_model()

    def test_a_folder_is_checked_against_the_listing_or_taken_offline(self, hub, tmp_path,
                                                                      monkeypatch):
        folder = tmp_path / "Breeze-TTS-2"
        for path, data in MODEL_FILES.items():
            (folder / path).parent.mkdir(parents=True, exist_ok=True)
            (folder / path).write_bytes(data)
        found = breeze.install_from("model", str(tmp_path))
        assert found.model_installed and hub.downloads == []
        marker = json.loads((paths.breeze_model_root(MODEL) / paths.INSTALLED_FILENAME)
                            .read_text(encoding="utf-8"))
        assert marker["commit"] == HUB_COMMIT and marker["source"] == "local"

        breeze.uninstall()

        def offline(url, ceiling=0, authorized=False, name=""):
            raise models.VoiceError("The publisher could not be asked.")

        monkeypatch.setattr(models, "read_json", offline)
        found = breeze.install_from("model", str(folder))
        assert found.model_installed
        marker = json.loads((paths.breeze_model_root(MODEL) / paths.INSTALLED_FILENAME)
                            .read_text(encoding="utf-8"))
        assert marker["commit"] == "" and marker["verified_by"]["config.json"] == \
            "your own files"
        assert "README.md" not in marker["bytes"]

    def test_an_edited_file_in_a_folder_is_refused(self, hub, tmp_path):
        folder = tmp_path / "model"
        for path, data in MODEL_FILES.items():
            (folder / path).parent.mkdir(parents=True, exist_ok=True)
            (folder / path).write_bytes(data)
        (folder / "audio_tokenizer" / "model.safetensors").write_bytes(_safetensors() + b"\x07")
        with pytest.raises(breeze.BreezeError, match="contents are not what"):
            breeze.install_from("model", str(folder))


# --------------------------------------------------------------------------- #
# Status, settings and what a render needs
# --------------------------------------------------------------------------- #


class TestStatus:
    def test_an_unsupported_machine_is_told_what_it_needs(self, voice_root, monkeypatch):
        monkeypatch.setattr(models, "current_platform", lambda: ("linux", "x86_64", "3.12"))
        found = breeze.status()
        assert not found.supported and not found.ready
        assert "Windows with an NVIDIA card on Python 3.13" in found.message
        assert "VibeVoice" in found.message, "the other engine is unaffected, and said so"

    def test_a_supported_machine_starts_with_nothing_and_says_how_much(self, windows):
        found = breeze.status()
        assert found.supported and not found.runtime_installed and not found.model_installed
        assert found.message == "Setup required — Breeze's runtime and model still to install."
        assert "Flash attention 2" in found.runtime_message
        assert "non-commercial" in found.model_message
        assert found.download_bytes == found.runtime_bytes + found.model_bytes > 0

    def test_the_page_gets_the_engines_vocabulary_and_its_licence(self, windows):
        found = breeze.public_status()
        assert (found["id"], found["label"], found["max_takes"]) == ("breeze", "Breeze TTS 2", 4)
        assert [field["key"] for field in found["fields"]] == \
            ["cfg_scale", "attention", "max_new_tokens", "line_gap_ms", "sampling",
             "temperature", "top_p"]
        assert found["capabilities"]["design"] is True
        assert "Non-Commercial" in found["license"]
        assert [part["id"] for part in found["parts"]] == ["runtime", "model"]


class TestSettings:
    def test_the_defaults(self, voice_root):
        assert breeze.settings() == {"card_uuid": "", "model_id": MODEL, "fast_path": False,
                                     "keep_warm": True}

    def test_values_are_validated_and_persisted(self, voice_root):
        found = breeze.set_settings({"fast_path": "on", "keep_warm": 0,
                                     "card_uuid": "GPU-1234-abcd"})
        assert (found["fast_path"], found["keep_warm"], found["card_uuid"]) == \
            (True, False, "GPU-1234-abcd")
        assert breeze.settings() == found

    @pytest.mark.parametrize("values, words", [
        ({"steps": 10}, "not a Breeze setting"),
        ({"model_id": "vibevoice-7b"}, "not a Breeze model"),
        ({"card_uuid": "GPU-1;rm -rf"}, "not a card identifier"),
        ({"fast_path": "maybe"}, "on or off"),
    ])
    def test_out_of_range_and_unknown_values_are_refused(self, voice_root, values, words):
        with pytest.raises(breeze.BreezeError, match=words):
            breeze.set_settings(values)

    def test_a_stored_value_that_no_longer_validates_falls_back_alone(self, voice_root):
        breeze._settings_write({"fast_path": True, "card_uuid": "bad;card", "model_id": "x"})
        assert breeze.settings() == {"card_uuid": "", "model_id": MODEL, "fast_path": True,
                                     "keep_warm": True}


class TestWhatARenderNeeds:
    def test_the_fast_path_asks_for_its_graphs_and_takes_ask_for_nothing_more(self, voice_root):
        eager = breeze.need_vram_bytes(MODEL)
        assert eager == breeze.WEIGHTS_BYTES_DEFAULT + breeze.WORKING_BYTES_DEFAULT
        assert breeze.need_vram_bytes(MODEL, takes=4) == eager, "takes run one after another"
        assert breeze.need_vram_bytes(MODEL, fast_path=True) == eager + breeze.FAST_BYTES_DEFAULT
        breeze.set_settings({"fast_path": True})
        assert breeze.need_vram_bytes(MODEL) == eager + breeze.FAST_BYTES_DEFAULT

    def test_each_path_keeps_its_own_measurement(self, voice_root):
        breeze.note_peak(MODEL, 30 * 1024 ** 3, takes=3, fast_path=True)
        breeze.note_peak(MODEL, 2 * 1024 ** 3)
        measured = breeze.calibration()
        assert measured[f"{MODEL}+fast"]["peak_bytes"] == 30 * 1024 ** 3, "a take's own peak"
        assert measured[MODEL]["peak_bytes"] == 2 * 1024 ** 3
        assert breeze.need_vram_bytes(MODEL, fast_path=True) == 30 * 1024 ** 3
        assert breeze.need_vram_bytes(MODEL, fast_path=False) == \
            breeze.WEIGHTS_BYTES_DEFAULT + breeze.WORKING_BYTES_DEFAULT
        breeze.note_peak("../escape", 1)
        breeze.note_peak(MODEL, "lots")
        assert set(breeze.calibration()) == {MODEL, f"{MODEL}+fast"}

    def test_ram_stands_on_the_estimate_until_a_render_measures_it(self, voice_root):
        assert breeze.need_ram_bytes(MODEL) == breeze.RAM_BYTES_DEFAULT
        breeze.note_peak(MODEL, 1, rss_bytes=12 * 1024 ** 3, fast_path=True)
        assert breeze.need_ram_bytes(MODEL) == 12 * 1024 ** 3


class TestTheWorkersSide:
    def test_it_names_the_card_its_code_and_runs_offline_with_no_credential(
            self, voice_root, monkeypatch):
        monkeypatch.setenv("HF_TOKEN", "hf_secret")
        found = breeze.worker_environment("GPU-0123abcd-4567-89ef-0123-456789abcdef")
        assert found["CUDA_VISIBLE_DEVICES"] == "GPU-0123abcd-4567-89ef-0123-456789abcdef"
        assert found[worker.SOURCE_VARIABLE] == str(paths.breeze_source_root())
        assert (found["HF_HUB_OFFLINE"], found["TRANSFORMERS_OFFLINE"]) == ("1", "1")
        assert not any(name in found for name in models.CREDENTIAL_VARIABLES)
        assert "hf_secret" not in json.dumps(found)
        assert "CUDA_VISIBLE_DEVICES" not in breeze.worker_environment("")
        with pytest.raises(breeze.BreezeError):
            breeze.worker_environment("0 && calc")

    def test_the_worker_script_and_model_directory_are_this_extensions(self, voice_root):
        assert breeze.worker_script() == ROOT / "breeze_worker" / "worker.py"
        assert breeze.model_dir() == paths.breeze_model_root(MODEL)
        assert paths.breeze_source_root() == paths.breeze_runtime_root() / "breeze-tts"


class TestTheVocabulary:
    def test_a_new_configuration_starts_where_upstream_and_the_owner_said(self):
        assert breeze.defaults() == {"cfg_scale": 4.0, "attention": "flash_attention_2",
                                     "max_new_tokens": None, "line_gap_ms": 150,
                                     "sampling": True, "temperature": 0.9, "top_p": 1.0}
        assert breeze.SAMPLE_RATE == worker.SAMPLE_RATE and breeze.MAX_TAKES == worker.MAX_TAKES
        assert breeze.CFG_MAX == worker.CFG_MAX and breeze.MAX_NEW_TOKENS == worker.MAX_NEW_TOKENS
        assert set(breeze.ATTENTION_NAMES) == set(worker.ATTENTION)

    def test_cleaning_holds_every_bound_the_worker_holds(self):
        assert breeze.clean({"cfg_scale": "6.5", "line_gap_ms": 0})["cfg_scale"] == 6.5
        for values in ({"cfg_scale": 8.5}, {"cfg_scale": 0.5}, {"max_new_tokens": 1501},
                       {"line_gap_ms": 2001}, {"attention": "flash_attention_3"},
                       {"top_p": 0}):
            with pytest.raises(breeze.BreezeError):
                breeze.clean(values)

    def test_the_capabilities_are_breezes(self):
        found = breeze.capabilities()
        assert (found["speakers"], found["takes"], found["design"], found["directions"]) == \
            ("per_line", "sequential", True, True)
        assert found["voice"] == {"sample": True, "transcript": True, "description": True,
                                  "instruction": True}
        assert "(laugh)" in found["events"]["en"] and "[笑]" in found["events"]["zh"]
