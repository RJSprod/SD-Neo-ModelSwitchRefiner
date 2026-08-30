"""The recording cleanup engine: DeepFilterNet, on an interpreter of its own.

Optional, installed and removed on its own, started only while a recording is
being cleaned and stopped as soon as it is idle. It is not a text-to-speech
engine and is not part of the engine selector (I-1): it takes a WAV and gives
back a quieter one, and Kokoro and Sopro neither know nor care that it exists.

Why a third runtime
-------------------
DeepFilterNet's inference path imports Torch, and its DSP -- the ERB filterbank
and the deep-filtering stage -- lives in a Rust extension, ``DeepFilterLib``,
whose newest published wheels are for CPython 3.11. Forge runs on whatever
Python it was installed under, 3.13 on the machine this was written for. So this
engine can share neither the host's interpreter nor Sopro's cp313 Torch, and it
brings both of its own: about 242 MB, most of it a second Torch.

That is a lot to clean a twenty-second clip and it is stated wherever the user
is asked to agree to it. The alternative considered and rejected was a denoiser
that gates on a speech classifier; the measurement behind that decision is in
``docs/17-voice-chat-sopro.md``.

The interpreter is the one unpinned executable
----------------------------------------------
Every wheel below is checked against a SHA-256 committed in this repository. The
interpreter is not: python.org is not reachable from the workspace the manifest
was generated in, so its digest is recorded on the machine that first fetches it
rather than checked against a constant somebody reviewed. That is weaker than
everything around it, it was agreed to explicitly, and it is said out loud here
rather than left for somebody to notice.

What it never does
------------------
It never runs while nothing is being cleaned, it never touches a graphics
device, and it never outlives the WebUI -- the same five doors Sopro's runtime
holds, in ``mc_voice_cleanup_runtime``.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path

import mc_voice_models as models
import mc_voice_paths as paths

logger = logging.getLogger("model_chain")
"""Handler is attached once, in mc_memory."""

ENGINE = "deepfilternet"
LABEL = "DeepFilterNet3"

KIND = "cleanup"
"""The key this engine's install progress is filed under.

Beside ``runtime``, ``stt``, ``tts`` and ``sopro`` in the one progress map, so
the settings row draws it with the same code and two installs cannot run at
once under the same name."""

INTRAOP_THREADS = 4
"""The same budget Sopro and Kokoro get, so nothing here can take the machine
over while a reply is being spoken."""


class CleanupError(RuntimeError):
    """A cleanup operation that could not be completed. Never fatal."""


@dataclass(frozen=True)
class Status:
    runtime_ready: bool
    model_ready: bool
    platform_supported: bool
    runtime_message: str
    model_message: str
    message: str
    download_bytes: int

    @property
    def ready(self) -> bool:
        return self.runtime_ready and self.model_ready


# --------------------------------------------------------------------------- #
# The manifest
# --------------------------------------------------------------------------- #


def manifest() -> dict:
    """The checked-in trust root for everything this engine may fetch."""
    path = paths.cleanup_manifest_path()
    try:
        found = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise CleanupError(f"The cleanup manifest could not be read ({exc}).") from None
    if not isinstance(found, dict) or int(found.get("schema") or 0) != 1:
        raise CleanupError("The cleanup manifest is a newer schema than this build reads.")
    return found


def supported_platform() -> bool:
    """Windows x86-64 only, and for the reason Sopro is.

    PyPI's Linux Torch wheels pull the whole CUDA closure through their
    dependency markers, and this engine neither claims nor consumes a graphics
    device. A Linux CPU closure comes from PyTorch's own ``+cpu`` index and is a
    separate, deliberate addition.
    """
    system, machine, _python = models.current_platform()
    return system == "windows" and machine in ("amd64", "x86_64")


def _wheels() -> list:
    return list((manifest().get("runtime") or {}).get("wheels") or ())


def download_bytes() -> int:
    found = manifest()
    runtime = found.get("runtime") or {}
    return (int(runtime.get("bytes") or 0)
            + int((runtime.get("interpreter") or {}).get("about_bytes") or 0)
            + int((found.get("model") or {}).get("about_bytes") or 0))


def _artifacts() -> list:
    """Every wheel as a :class:`mc_voice_models.Artifact`, plus the interpreter.

    The interpreter is last and carries no hash, which is what makes
    :func:`mc_voice_models._resolve` ask the publisher about it and
    :func:`mc_voice_models._record` write down what arrived.
    """
    found = manifest()
    runtime = found.get("runtime") or {}
    made = [models.Artifact(filename=item["filename"], local_name=item["local_name"],
                            url=item["url"], size=int(item.get("bytes") or 0),
                            sha256=str(item.get("sha256") or "") or None)
            for item in runtime.get("wheels") or ()]
    interpreter = runtime.get("interpreter") or {}
    made.append(models.Artifact(
        filename=interpreter["filename"], local_name=interpreter["filename"],
        url=interpreter["url"], size=None, sha256=None))
    return made


def _model_artifact() -> "models.Artifact":
    found = (manifest().get("model") or {})
    return models.Artifact(filename=found["filename"], local_name=found["filename"],
                           url=found["url"], size=None, sha256=None)


# --------------------------------------------------------------------------- #
# What is installed
# --------------------------------------------------------------------------- #


def runtime_python() -> "Path | None":
    """The cp311 interpreter, or ``None`` when it is not installed."""
    found = paths.cleanup_runtime_root() / "env" / "python.exe"
    return found if found.exists() else None


def installed() -> dict:
    try:
        found = json.loads(paths.cleanup_runtime_manifest().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return found if isinstance(found, dict) else {}


def _closure_id() -> str:
    """A hash of every pinned digest, so a changed closure reads as stale.

    The interpreter is deliberately not in it: it has no committed digest to
    hash, and folding a recorded-on-this-machine value in would make the
    fingerprint mean something different on every installation.
    """
    import hashlib

    digest = hashlib.sha256()
    for item in _wheels():
        digest.update((str(item.get("sha256") or "") + "\n").encode("ascii"))
    return digest.hexdigest()[:16]


def model_ready() -> bool:
    root = paths.cleanup_model_root()
    expects = (manifest().get("model") or {}).get("expects") or ()
    return root.is_dir() and any((root / name).exists() for name in expects)


def status() -> Status:
    """What is installed, said the way the row draws it."""
    try:
        supported = supported_platform()
    except Exception:
        logger.debug("Model Chain: could not read the cleanup platform", exc_info=True)
        supported = False
    if not supported:
        system, machine, python = models.current_platform()
        return Status(False, False, False,
                      f"Not available on {system}/{machine}",
                      "Not available here",
                      f"Recording cleanup has no tested CPU runtime for {system}/{machine}. "
                      f"The in-page cleanup works everywhere and needs no download.",
                      0)

    record = installed()
    fresh = bool(record) and str(record.get("closure") or "") == _closure_id()
    model = False
    try:
        model = model_ready()
    except Exception:
        logger.debug("Model Chain: could not read the cleanup model", exc_info=True)
    size = 0
    try:
        size = download_bytes()
    except Exception:
        logger.debug("Model Chain: could not size the cleanup download", exc_info=True)

    runtime_message = (
        f"Installed — DeepFilterNet {record.get('deepfilternet') or '?'}, "
        f"Torch {record.get('torch') or '?'}, Python {record.get('python') or '?'}, CPU only"
        if fresh else
        ("Installed by an older build and needs installing again" if record
         else f"Not installed — about {models._bytes_label(size)}"))
    model_message = ("Installed — " + LABEL) if model else "Not installed"
    return Status(fresh, model, True, runtime_message, model_message,
                  runtime_message if not fresh else model_message, size)


# --------------------------------------------------------------------------- #
# Installing
# --------------------------------------------------------------------------- #


def install(on_status=None, on_progress=None) -> Status:
    """Fetch, build, prove and promote. A transaction, like every other here.

    Nothing outside the staging directory is touched until every declared byte
    has arrived, the interpreter runs, DeepFilterNet imports, and a second of
    audio has actually been through it. A failure anywhere leaves the machine
    exactly as it was.
    """
    say = models._narrator(KIND, on_status)
    tick = models._ticker(KIND, on_progress)

    with models._claim(KIND, say):
        return _install(say, tick)


def _install(say, tick) -> Status:
    if not supported_platform():
        raise CleanupError(status().message)

    staging = paths.cleanup_staging_root()
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True, exist_ok=True)
    try:
        say("Checking what the publishers say these files are…")
        artifacts = _artifacts()
        expectations = models._expectations(artifacts, say)
        # After the sizes are known rather than against a guess: the
        # interpreter has no pinned byte count and the publisher has just been
        # asked for one.
        models._make_room(artifacts, staging, expectations)
        digests = models._fetch_all(artifacts, staging, say, tick, 0.55, expectations)

        say("Building the isolated cleanup runtime…")
        _build_environment(staging)
        tick(0.7)

        say("Fetching DeepFilterNet's model…")
        model = _model_artifact()
        model_expectations = models._expectations([model], say)
        digests.update(models._fetch_all([model], staging, say, tick, 0.85,
                                         model_expectations))
        _unpack_model(staging / model.local_name, staging / "model")
        tick(0.9)

        say("Checking that DeepFilterNet runs on this machine…")
        report = _smoke_test(staging)
        _promote(staging, report, digests)
        tick(1.0)
    finally:
        shutil.rmtree(staging, ignore_errors=True)

    found = status()
    logger.info("Model Chain: the cleanup engine is installed — %s, DeepFilterNet %s, "
                "Torch %s", LABEL, installed().get("deepfilternet") or "?",
                installed().get("torch") or "?")
    say("Recording cleanup is installed.")
    return found


def _build_environment(staging: Path) -> None:
    """The embeddable interpreter, unpacked, with the closure beside it.

    No ``venv`` and no pip, because the embeddable distribution has neither --
    which suits an installer that has never used pip. What it does have is a
    ``python311._pth`` listing everything on ``sys.path``, and rewriting that
    file is how ``site-packages`` gets on it. ``import site`` is added back
    because Torch's own ``__init__`` reads ``site.getsitepackages`` on Windows
    to find its DLL directory.
    """
    environment = staging / "env"
    environment.mkdir(parents=True, exist_ok=True)
    interpreter_zip = staging / ((manifest().get("runtime") or {})
                                 .get("interpreter") or {})["filename"]
    if not interpreter_zip.is_file():
        raise CleanupError("The interpreter did not arrive. Nothing was installed.")
    try:
        with zipfile.ZipFile(interpreter_zip) as bundle:
            for member in bundle.infolist():
                name = member.filename.lstrip("./")
                if not name or name.endswith("/") or "\\" in name or ".." in name.split("/"):
                    continue
                where = (environment / name).resolve()
                try:
                    where.relative_to(environment.resolve())
                except ValueError:
                    continue
                where.parent.mkdir(parents=True, exist_ok=True)
                with bundle.open(member) as source, open(where, "wb") as handle:
                    shutil.copyfileobj(source, handle)
    except (OSError, zipfile.BadZipFile) as exc:
        raise CleanupError(f"The interpreter could not be unpacked ({exc}).") from None

    interpreter = environment / "python.exe"
    if not interpreter.exists():
        raise CleanupError("The interpreter archive contained no python.exe. Nothing was "
                           "installed.")

    target = environment / "site-packages"
    target.mkdir(parents=True, exist_ok=True)
    found = sorted(environment.glob("python*._pth"))
    if not found:
        raise CleanupError("The interpreter archive had no path file, so this build cannot "
                           "put the closure on its path. Nothing was installed.")
    lines = found[0].read_text(encoding="utf-8").splitlines()
    kept = [line for line in lines if line.strip() != "#import site"]
    found[0].write_text("\n".join(kept + ["site-packages", "import site"]) + "\n",
                        encoding="utf-8")

    for item in _wheels():
        wheel = staging / item["local_name"]
        if not wheel.is_file():
            raise CleanupError(f"{item['filename']} is missing from the staged download. "
                               f"Nothing was installed.")
        added = models._unpack_wheel(wheel, target)
        logger.info("Model Chain: cleanup unpacked %s (%s)", item["filename"],
                    ", ".join(added[:6]) or "nothing")
    for name in ("df", "torch", "libdf"):
        if not any(target.glob(name + "*")):
            raise CleanupError(f"The cleanup wheels were unpacked but {name} is not in "
                               f"{target}. Nothing was installed.")


def _unpack_model(archive: Path, destination: Path) -> None:
    """DeepFilterNet's own zip, expanded, refusing every escaping member."""
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    try:
        with zipfile.ZipFile(archive) as bundle:
            for member in bundle.infolist():
                name = member.filename.lstrip("./")
                if not name or name.endswith("/"):
                    continue
                # The publisher's archive has the model name as its top folder;
                # it is stripped so ``init_df`` gets the directory it expects.
                parts = [part for part in name.split("/") if part not in ("", ".", "..")]
                if len(parts) > 1:
                    parts = parts[1:]
                where = (destination / "/".join(parts)).resolve()
                try:
                    where.relative_to(root)
                except ValueError:
                    logger.warning("Model Chain: the cleanup model archive held a member "
                                   "that would have been written outside it")
                    continue
                where.parent.mkdir(parents=True, exist_ok=True)
                with bundle.open(member) as source, open(where, "wb") as handle:
                    shutil.copyfileobj(source, handle)
    except (OSError, zipfile.BadZipFile) as exc:
        raise CleanupError(f"The cleanup model could not be unpacked ({exc}).") from None
    expects = (manifest().get("model") or {}).get("expects") or ()
    if not any((destination / name).exists() for name in expects):
        raise CleanupError("The cleanup model archive did not contain a model. Nothing was "
                           "installed.")


def _smoke_test(staging: Path) -> dict:
    """Prove the staged runtime runs, imports, is CPU-only, and denoises.

    Before promotion, so an environment that builds and cannot run never becomes
    the installed one. A second of noise through the real model is the only
    check that means anything: an import that succeeds proves the wheels landed
    and nothing about whether the model loads.
    """
    import subprocess

    interpreter = staging / "env" / "python.exe"
    script = paths.cleanup_worker_script()
    environ = dict(os.environ)
    environ.update(worker_environment())
    try:
        found = subprocess.run(  # noqa: S603 - a path this module built
            [str(interpreter), str(script), "--selftest", "--model", str(staging / "model")],
            capture_output=True, text=True, timeout=600.0, env=environ,
            cwd=str(paths.extension_root()))
    except Exception as exc:
        raise CleanupError(f"The cleanup self-test could not be run "
                           f"({exc.__class__.__name__}: {exc}).") from None
    if found.returncode != 0:
        tail = (found.stderr or found.stdout or "").strip().splitlines()[-6:]
        logger.warning("Model Chain: the cleanup self-test failed with exit code %s.\n%s",
                       found.returncode, "\n".join(tail))
        raise CleanupError("DeepFilterNet was installed but could not run on this machine, "
                           "so nothing was kept.")
    try:
        report = json.loads((found.stdout or "").strip().splitlines()[-1])
    except (ValueError, IndexError):
        raise CleanupError("The cleanup self-test produced no report, so nothing was "
                           "kept.") from None
    if not report.get("ok"):
        raise CleanupError(str(report.get("error") or "The cleanup self-test did not pass."))
    logger.info("Model Chain: the cleanup self-test passed — DeepFilterNet %s, Torch %s, "
                "Python %s, %.1f s of audio in %d ms",
                report.get("deepfilternet"), report.get("torch"), report.get("python"),
                float(report.get("seconds") or 0.0), int(report.get("elapsed_ms") or 0))
    return report


def _promote(staging: Path, report: dict, digests: dict) -> None:
    """The last step, and the only one that touches the installed tree."""
    runtime = paths.cleanup_runtime_root()
    model = paths.cleanup_model_root()
    for target, source in ((runtime, staging / "env"), (model, staging / "model")):
        previous = target.with_name(target.name + ".old")
        shutil.rmtree(previous, ignore_errors=True)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            os.replace(target, previous)
        try:
            os.replace(source, target)
        except OSError:
            if previous.exists():
                os.replace(previous, target)
            raise CleanupError("The cleanup runtime could not be moved into place. Nothing "
                               "was changed.") from None
        shutil.rmtree(previous, ignore_errors=True)

    record = {
        "engine": ENGINE,
        "closure": _closure_id(),
        "python": report.get("python") or "",
        "torch": report.get("torch") or "",
        "deepfilternet": report.get("deepfilternet") or "",
        "digests": digests,
    }
    paths.cleanup_runtime_manifest().parent.mkdir(parents=True, exist_ok=True)
    paths.cleanup_runtime_manifest().write_text(json.dumps(record, indent=2) + "\n",
                                                encoding="utf-8")


def uninstall() -> None:
    """Take it all away. The recordings and voices it cleaned are untouched."""
    import mc_voice_cleanup_runtime as runtime

    try:
        runtime.shutdown()
    except Exception:
        logger.debug("Model Chain: could not stop the cleanup engine before removing it",
                     exc_info=True)
    for target in (paths.cleanup_runtime_root(), paths.cleanup_model_root(),
                   paths.cleanup_staging_root()):
        shutil.rmtree(target, ignore_errors=True)
    logger.info("Model Chain: the cleanup engine was removed")


def worker_environment() -> dict:
    """The environment the cleanup worker runs in. CPU only, enforced here.

    The same blunt instrument Sopro's runtime uses: a Torch build that would
    happily have found a graphics device finds no devices to enumerate, so this
    cannot take VRAM from an image generation while somebody tidies a recording.
    """
    return {
        "CUDA_VISIBLE_DEVICES": "",
        "HIP_VISIBLE_DEVICES": "",
        "ROCR_VISIBLE_DEVICES": "",
        "PYTORCH_NO_CUDA_MEMORY_CACHING": "1",
        "OMP_NUM_THREADS": str(INTRAOP_THREADS),
        "MKL_NUM_THREADS": str(INTRAOP_THREADS),
        "OPENBLAS_NUM_THREADS": str(INTRAOP_THREADS),
        "PYTHONNOUSERSITE": "1",
        "PYTHONUNBUFFERED": "1",
        "DF_LOG_LEVEL": "WARNING",
    }
