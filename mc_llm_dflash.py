"""The DFlash2 llama.cpp family: installed apart, proved apart, lost apart.

DFlash2 is not in llama.cpp. At the time this was written it is pull request
27342, open, and the publisher of the Qwen 3.8 weights pins its own testing to
one commit of that branch. So the build that can run the speculative sidecar is
a *different program* from the build everything else in this extension runs on,
and the whole of this module follows from one rule about that:

    installing, failing to install, or losing the special runtime may not
    change the ordinary one in any way.

Which is why it is a runtime *family* rather than an upgrade.
:mod:`mc_llm_setup` already had the mechanism -- a directory per family, with a
``.runtime-id`` naming what is in it -- because a 3090 role and a 5090 role can
need different CUDA builds. This adds a family to that arrangement instead of
inventing a second one, so an installation with a DFlash2 build has one more
directory and an installation without one has exactly what it always had.

Two routes in, and the second is the one that works today
---------------------------------------------------------
``download`` fetches a pinned archive from ``dflash2-runtimes.json``, which is
a checked-in trust root of the same kind the model registry is: a URL and a
SHA-256 that were reviewed rather than discovered. An entry with no digest is
one nobody has published a build of, and it is *refused* rather than guessed
at -- there is no route through this module that fetches something whose bytes
are not named in this repository.

While the pull request is open there may well be no such archive, and the
realistic route is ``adopt``: build the branch at the pinned commit, point
Setup at the directory, and the build is copied into its own family with a
provenance file recording which commit it claims to be. That claim is
*recorded*, not trusted -- what decides whether DFlash2 is offered is the smoke
test below.

Why ``--help`` is not proof
---------------------------
Upstream llama.cpp already carries DFlash terminology, so a build that has
never heard of this pull request can advertise ``--spec-type`` and even print
``draft-dflash`` in its help. Treating that as compatibility is how a user gets
a server that starts, produces nothing usable, and is labelled Lightning. So
the capability record here is written from a real load of the real Qwen 3.8
target with the real Blackfrost sidecar, answering a real request -- and
:data:`Capability.text` and :data:`Capability.vision` are two fields because
the pull request's multimodal support arrived separately from its text support
and one passing says nothing about the other.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import mc_llm_files
import mc_llm_paths
import mc_llm_setup

logger = logging.getLogger("model_chain")
"""Handler is attached once, in mc_memory."""

MANIFEST_PATH = (Path(__file__).resolve().parent / "prompt_master" / "models"
                 / "dflash2-runtimes.json")

PROVENANCE_MARKER = ".dflash2-id"
"""Written inside an installed family directory, naming what it holds.

Beside ``mc_llm_setup``'s own ``.runtime-id`` rather than instead of it: that
file answers "which family may install here", which is a question about
directories, and this one answers "which llama.cpp commit is this", which is a
question about provenance. Conflating them would mean a build adopted from a
folder and a build extracted from a pinned archive could not be told apart.
"""

CAPABILITY_FILENAME = "dflash2-capabilities.json"
"""Where the smoke-test results live, keyed by component id.

In the data directory rather than in the runtime directory, so that replacing a
build does not silently inherit the previous build's proof -- the record
carries the executable's own fingerprint and a record whose fingerprint no
longer matches is treated as absent. That is the difference between "this
runtime passed" and "a runtime that used to be at this path passed".
"""

SMOKE_PROMPT = "Reply with exactly READY and nothing else."
SMOKE_ANSWER = "READY"
SMOKE_TOKENS = 8
SMOKE_TIMEOUT = 300.0
"""The text smoke test: deterministic, semantic, and short.

Semantic because ``/health`` answering proves a file was opened and nothing
else. A server that has loaded a model whose architecture the build does not
really support reaches ``/health`` and then returns punctuation, an empty
string, or a loop -- so what is asked for is a specific word at temperature
zero, and getting it back is the smallest thing that proves the target, the
draft, the template and the speculative path all work together.
"""

VISION_SMOKE_PROMPT = "Answer with one word: what colour is this image?"
VISION_SMOKE_TOKENS = 8
VISION_SMOKE_COLOUR = (32, 96, 224)
"""A four-pixel picture of one flat colour, and a question about it.

Not an accuracy test -- a small vision model naming this "blue" rather than
"azure" is a pass, and so is "a solid colour". What is being proved is that the
projector loaded under the DFlash2 build and that an image request reached the
model at all, which is exactly the thing the pull request's multimodal work
kept changing during review.
"""


class DFlashError(RuntimeError):
    """Something a user can act on, phrased as a sentence for the panel."""


# --------------------------------------------------------------------------- #
# The checked-in trust root
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Build:
    """One pinned DFlash2 build, published or not."""

    family: str
    component_id: str
    label: str
    cuda: str
    architectures: tuple[int, ...]
    requires_component: str
    """Which ordinary runtime family this one stands beside, never replaces."""
    archive: str
    commit: str
    pull_request: str
    url: str = ""
    sha256: str = ""
    size: int | None = None

    @property
    def published(self) -> bool:
        """Whether this repository names bytes that can actually be fetched.

        Both halves, because either alone is useless: a URL with no digest is a
        download nobody reviewed, and a digest with no URL is a promise about a
        file with no source. An unpublished entry is a complete, honest
        description of a build somebody has to make themselves.
        """
        return bool(self.url) and len(self.sha256) == 64

    @property
    def short_commit(self) -> str:
        return self.commit[:7]

    def describe(self) -> str:
        return (f"{self.label} · llama.cpp {self.short_commit} · "
                f"sm_{'/sm_'.join(str(arch) for arch in self.architectures)}")


def builds() -> list[Build]:
    """Every DFlash2 build this extension knows how to install. Never raises.

    An unreadable manifest is an empty list and a warning, not an exception:
    the Setup panel renders this, and a broken optional accelerator must not be
    able to stop somebody configuring an ordinary model.
    """
    try:
        document = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        if not isinstance(document, dict):
            raise ValueError("the DFlash2 manifest is not a JSON object")
    except (OSError, ValueError):
        logger.warning("Model Chain: the DFlash2 runtime manifest could not be read",
                       exc_info=True)
        return []

    commit = str(document.get("commit") or "")
    pull_request = str(document.get("pull_request") or "")
    found: list[Build] = []
    for raw in document.get("families") or []:
        if not isinstance(raw, dict):
            continue
        url = str(raw.get("url") or "")
        if url and not url.startswith("https://"):
            logger.warning("Model Chain: a DFlash2 runtime entry names a non-HTTPS URL and "
                           "was ignored")
            continue
        found.append(Build(
            family=str(raw.get("family") or ""),
            component_id=str(raw.get("component_id") or ""),
            label=str(raw.get("label") or raw.get("component_id") or ""),
            cuda=str(raw.get("cuda") or ""),
            architectures=tuple(int(value) for value in raw.get("compute_architectures") or ()),
            requires_component=str(raw.get("requires_component") or ""),
            archive=str(raw.get("archive") or ""),
            commit=commit,
            pull_request=pull_request,
            url=url,
            sha256=str(raw.get("sha256") or "").strip().casefold(),
            size=raw.get("size"),
        ))
    return [build for build in found if build.component_id]


def build_for(component_id: str) -> Build | None:
    """The build ``component_id`` names, or ``None``."""
    wanted = str(component_id or "").strip()
    return next((build for build in builds() if build.component_id == wanted), None)


def build_for_device(device) -> Build | None:
    """Which DFlash2 build a card should get, or ``None`` for a processor.

    The same question :func:`prompt_master.inference.device_detection.
    runtime_component_id` answers for the ordinary runtime, asked of this
    manifest instead -- so the answer is the CUDA 13 build for a Blackwell card
    and, where a driver constrains it, the Ampere-only CUDA 12 one. There is no
    processor build and there is not going to be: DFlash2 is a CUDA path, and a
    machine with no card has nothing for it to accelerate.
    """
    if device is None or getattr(device, "is_cpu", False):
        return None
    try:
        from prompt_master.inference.device_detection import runtime_component_id
    except Exception:
        return None
    ordinary = runtime_component_id(device)
    available = builds()
    matched = next((build for build in available
                    if build.requires_component == ordinary), None)
    # No entry standing beside this device's ordinary runtime is not a reason
    # to install nothing: a CUDA 13 build targets every architecture the CUDA
    # 12 one does, so it is the safe direction to fall back in.
    return matched or next((build for build in available if build.cuda.startswith("13")), None)


# --------------------------------------------------------------------------- #
# What is installed
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Provenance:
    """What an installed family directory says about itself."""

    component_id: str = ""
    commit: str = ""
    source: str = ""
    """``"pinned archive"`` or the directory a local build was adopted from."""
    installed_at: float = 0.0

    @property
    def known(self) -> bool:
        return bool(self.component_id)


def runtime_directory(component_id: str, root=None) -> Path:
    """Where ``component_id`` lives, which is never the ordinary runtime's home.

    :func:`mc_llm_setup.runtime_directory` hands the plain ``runtime/``
    directory to the first family installed, which on a fresh machine could be
    this one -- and a DFlash2 build sitting in the directory every ordinary
    lookup falls back to is precisely the overwrite this module exists to
    prevent. So the family name is always used here.
    """
    base = Path(root) if root is not None else mc_llm_paths.data_root()
    return base / mc_llm_setup.family_dirname(component_id)


def executable(component_id: str, root=None) -> Path | None:
    """The DFlash2 llama-server for ``component_id``, or ``None``."""
    directory = runtime_directory(component_id, root)
    if not directory.is_dir():
        return None
    if mc_llm_setup.family_in(directory) not in ("", str(component_id)):
        return None
    return mc_llm_setup.server_in(directory)


def provenance(component_id: str, root=None) -> Provenance:
    """What the installed family records about where it came from."""
    directory = runtime_directory(component_id, root)
    try:
        document = json.loads((directory / PROVENANCE_MARKER).read_text(encoding="utf-8"))
        if not isinstance(document, dict):
            raise ValueError("provenance is not a JSON object")
    except (OSError, ValueError):
        return Provenance()
    return Provenance(
        component_id=str(document.get("component_id") or ""),
        commit=str(document.get("commit") or ""),
        source=str(document.get("source") or ""),
        installed_at=float(document.get("installed_at") or 0.0),
    )


def installed(component_id: str = "", root=None) -> Path | None:
    """The DFlash2 server this installation has, whichever family it is.

    ``component_id`` empty asks the broader question the panel asks -- "is
    there one at all" -- and answers with the first family that has a server
    in it, in manifest order.
    """
    if component_id:
        return executable(component_id, root)
    for build in builds():
        found = executable(build.component_id, root)
        if found is not None:
            return found
    return None


def installed_component(root=None) -> str:
    """Which family id is installed, or ``""``."""
    for build in builds():
        if executable(build.component_id, root) is not None:
            return build.component_id
    return ""


# --------------------------------------------------------------------------- #
# The capability record
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Capability:
    """What a DFlash2 runtime has actually been proved to do, here, once.

    Two independent booleans and not one, which is the whole point of the
    record. The pull request's multimodal support was reworked during review
    while its text path was stable, so a build can be entirely trustworthy for
    text and produce nothing usable for an image -- and a single "supported"
    flag would have to choose which of those two lies to tell.
    """

    component_id: str = ""
    text: bool = False
    vision: bool = False
    commit: str = ""
    fingerprint: str = ""
    checked_at: float = 0.0
    detail: str = ""
    vision_detail: str = ""
    """Why vision is where it is, kept apart from the text result's sentence.

    One field would make the two results share a reason, and the reason for
    "text passed" is never the reason for "vision did not".
    """

    @property
    def known(self) -> bool:
        return bool(self.checked_at)

    def describe(self) -> str:
        if not self.known:
            return "not verified yet"
        if not self.text:
            return "verification failed"
        return "text verified · vision verified" if self.vision else "text verified"


def _capability_path() -> Path:
    return mc_llm_paths.data_root() / "data" / CAPABILITY_FILENAME


def _fingerprint(path: Path) -> str:
    """A cheap identity for an executable: its size and modification time.

    Not a digest. Hashing a 200 MB binary on every panel draw is not something
    to do for a question whose only job is to notice that the file *changed* --
    and a user who replaces a build in place has changed both of these.
    """
    try:
        stat = path.stat()
    except OSError:
        return ""
    return f"{int(stat.st_size)}:{int(stat.st_mtime)}"


def capability(component_id: str, root=None) -> Capability:
    """What was recorded for ``component_id``, or an empty record.

    A record whose fingerprint no longer matches the executable is empty rather
    than stale: somebody who dropped a different build into the same directory
    has not inherited the previous one's proof, and the honest state for a
    runtime nobody has tested is "not verified yet".
    """
    found = executable(component_id, root)
    if found is None:
        return Capability(component_id=str(component_id or ""))
    try:
        document = json.loads(_capability_path().read_text(encoding="utf-8"))
        entry = (document or {}).get(str(component_id)) or {}
        if not isinstance(entry, dict):
            raise ValueError("a capability entry is not a JSON object")
    except (OSError, ValueError):
        return Capability(component_id=str(component_id or ""))

    recorded = Capability(
        component_id=str(component_id or ""),
        text=bool(entry.get("dflash2_text", False)),
        vision=bool(entry.get("dflash2_vision", False)),
        commit=str(entry.get("commit") or ""),
        fingerprint=str(entry.get("fingerprint") or ""),
        checked_at=float(entry.get("checked_at") or 0.0),
        detail=str(entry.get("detail") or ""),
        vision_detail=str(entry.get("vision_detail") or ""),
    )
    if recorded.fingerprint and recorded.fingerprint != _fingerprint(found):
        logger.info("Model Chain: the DFlash2 runtime at %s has changed since it was "
                    "verified; it will have to pass the smoke test again", found)
        return Capability(component_id=str(component_id or ""))
    return recorded


def record_capability(component_id: str, *, text: bool | None = None,
                      vision: bool | None = None, detail: str | None = None,
                      vision_detail: str | None = None, root=None) -> Capability:
    """Write one smoke-test result. Returns the record as it now stands.

    The two results are written independently and either may be left alone by
    passing ``None`` -- a vision test that ran after a text test must not be
    able to overwrite the text result with a default.

    Writing a text result always clears the vision one, in both directions and
    deliberately. ``False`` clears it because vision without text is not a state
    a runtime can be in; ``True`` clears it because a fresh text verification is
    a fresh verification, and carrying an image result forward across it would
    let a run that never sent an image leave "vision verified" on screen.
    """
    found = executable(component_id, root)
    if found is None:
        raise DFlashError(f"There is no DFlash2 runtime installed as {component_id}.")

    path = _capability_path()
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(document, dict):
            document = {}
    except (OSError, ValueError):
        document = {}

    entry = dict(document.get(str(component_id)) or {})
    if text is not None:
        entry["dflash2_text"] = bool(text)
        entry["dflash2_vision"] = False
        entry["vision_detail"] = ""
    if vision is not None:
        entry["dflash2_vision"] = bool(vision)
    if detail is not None:
        entry["detail"] = str(detail)
    if vision_detail is not None:
        entry["vision_detail"] = str(vision_detail)
    entry["commit"] = provenance(component_id, root).commit
    entry["fingerprint"] = _fingerprint(found)
    entry["checked_at"] = time.time()
    document[str(component_id)] = entry

    path.parent.mkdir(parents=True, exist_ok=True)
    from prompt_master.core.config import atomic_write_json

    atomic_write_json(path, document)
    return capability(component_id, root)


def forget_capability(component_id: str) -> None:
    """Drop a recorded result, so the runtime has to prove itself again."""
    path = _capability_path()
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(document, dict) or str(component_id) not in document:
            return
    except (OSError, ValueError):
        return
    document.pop(str(component_id), None)
    from prompt_master.core.config import atomic_write_json

    try:
        atomic_write_json(path, document)
    except OSError:
        logger.debug("Model Chain: could not clear the DFlash2 capability record",
                     exc_info=True)


# --------------------------------------------------------------------------- #
# Installing
# --------------------------------------------------------------------------- #


def adopt(source: str | Path, component_id: str = "") -> tuple[Path, str]:
    """Copy a locally built DFlash2 llama.cpp into its own family directory.

    The route that works while the pull request is open. ``source`` is the
    directory a ``cmake --build`` produced, or the ``llama-server`` inside it,
    and the whole directory is what is taken: section 7 of the design intent
    asks for the complete runtime distribution rather than one executable,
    because a server without the ggml and CUDA libraries beside it is a file
    that runs on the machine that built it and nowhere else.

    Nothing about the ordinary runtime is read or written on any path through
    this function, including the failing ones. That is the rule this module
    exists for, and it is arranged structurally rather than carefully: the
    destination is always a family directory, and :func:`runtime_directory`
    has no branch that can return ``runtime/``.
    """
    chosen = mc_llm_files.to_path(source)
    if chosen is None:
        raise DFlashError("Enter the path to the DFlash2 llama.cpp build, or to the folder "
                          "holding llama-server.")
    if not chosen.exists():
        raise DFlashError(f"There is nothing at {chosen}")

    found = mc_llm_setup.server_in(chosen) if chosen.is_dir() else chosen
    if found is None or found.name not in mc_llm_setup.SERVER_NAMES:
        raise DFlashError(
            f"{chosen} contains no llama-server executable. Point this at the build "
            f"directory of llama.cpp pull request 27342, or at the llama-server inside it."
        )

    directory = found.resolve().parent
    if component_id:
        build = build_for(component_id)
        if build is None:
            raise DFlashError(
                f"{component_id!r} is not a DFlash2 runtime family this extension knows "
                f"about, so there is nowhere to install one. The ordinary runtime is "
                f"unaffected.")
    else:
        build = _default_build()
        if build is None:
            raise DFlashError("This extension's DFlash2 manifest names no runtime families, "
                              "so there is nowhere to install one. The ordinary runtime is "
                              "unaffected.")

    destination = runtime_directory(build.component_id)
    incoming = destination.with_name(f".{destination.name}.incoming")
    shutil.rmtree(incoming, ignore_errors=True)
    try:
        mc_llm_setup.stage_build(directory, incoming)
        if mc_llm_setup.server_in(incoming) is None:
            raise DFlashError(f"{directory} contains no llama-server executable")
        _stamp(incoming, build, source=str(directory))
        mc_llm_setup.swap_in(incoming, destination)
    finally:
        shutil.rmtree(incoming, ignore_errors=True)

    placed = mc_llm_setup.server_in(destination)
    if placed is None:
        raise DFlashError(f"The DFlash2 build could not be read back from {destination}.")
    # A new build has not proved anything, whatever the one it replaced proved.
    forget_capability(build.component_id)
    logger.info("Model Chain: adopted a DFlash2 llama.cpp build from %s into %s",
                directory, destination)
    return placed, (
        f"Copied the DFlash2 build from {directory} into {destination}. It is recorded as "
        f"llama.cpp {build.short_commit}; verify it before Lightning is offered — a build "
        f"whose help text mentions DFlash is not the same thing as a build that runs it."
    )


def download(component_id: str = "", on_status=None, on_progress=None) -> Path:
    """Fetch and extract a pinned DFlash2 archive, when one has been published.

    The refusal is the interesting half. There is no route through this
    function that downloads something whose SHA-256 is not in this repository:
    an entry with no digest is not a build to be fetched cautiously, it is a
    build nobody has reviewed, and what comes back is a sentence naming the
    other route rather than a request.
    """
    from prompt_master.provisioning.downloader import download as fetch
    from prompt_master.provisioning.extractor import extract_zips_atomic
    from prompt_master.provisioning.manifest import Component

    say = on_status or (lambda _text: None)
    tick = on_progress or (lambda _fraction: None)

    build = build_for(component_id) if component_id else _default_build()
    if build is None:
        raise DFlashError("This extension's DFlash2 manifest names no runtime families.")
    if not build.published:
        raise DFlashError(
            f"{build.label} has no published archive in this extension yet — DFlash2 is "
            f"llama.cpp pull request {build.pull_request.rsplit('/', 1)[-1] or '27342'} and "
            f"is not part of any release. Build it at commit {build.short_commit} and point "
            f"Setup at the result; it is verified and recorded exactly as a pinned download "
            f"would be. The ordinary runtime is unaffected either way."
        )

    paths = mc_llm_paths.app_paths()
    paths.create_managed_dirs()
    component = Component(component_id=build.component_id, url=build.url,
                          destination=f"cache/downloads/{build.archive}",
                          size=build.size, sha256=build.sha256, version=build.commit)
    try:
        component.validate()
    except ValueError as exc:
        raise DFlashError(f"The pinned DFlash2 runtime is not usable: {exc}") from None

    say(f"Downloading {build.label}…")
    archive = fetch(component, paths.contained(component.destination),
                    lambda done, total: tick(0.9 * done / max(total, 1)))

    say("Extracting the DFlash2 runtime…")
    destination = runtime_directory(build.component_id)
    incoming = destination.with_name(f".{destination.name}.incoming")
    shutil.rmtree(incoming, ignore_errors=True)
    try:
        extract_zips_atomic([archive], incoming)
        if mc_llm_setup.server_in(incoming) is None:
            raise DFlashError("The DFlash2 archive contains no llama-server executable")
        _stamp(incoming, build, source="pinned archive")
        mc_llm_setup.swap_in(incoming, destination)
    finally:
        shutil.rmtree(incoming, ignore_errors=True)

    placed = mc_llm_setup.server_in(destination)
    if placed is None:
        raise DFlashError("The DFlash2 archive contains no llama-server executable")
    forget_capability(build.component_id)
    tick(1.0)
    return placed


def remove(component_id: str) -> None:
    """Delete one DFlash2 family. Ordinary llama.cpp is not touched.

    Offered because the design intent asks for it in as many words: losing the
    special runtime must break nothing, and the way to be sure of that is to be
    able to lose it on purpose.
    """
    build = build_for(component_id)
    if build is None:
        return
    shutil.rmtree(runtime_directory(build.component_id), ignore_errors=True)
    forget_capability(build.component_id)


def _default_build() -> Build | None:
    """Whichever build suits the configured device, or the first published one."""
    try:
        found = build_for_device(mc_llm_setup.configured_device())
    except Exception:
        found = None
    return found or next(iter(builds()), None)


def _stamp(directory: Path, build: Build, source: str) -> None:
    """Write both markers into a staged directory, before it is swapped in.

    Before rather than after, for the reason ``mc_llm_setup.download`` stamps
    its own marker before the swap: a directory is never in place without the
    file that says what is in it, so there is no window in which the next
    install reads an unmarked directory and decides it may overwrite this one.
    """
    from prompt_master.core.config import atomic_write_json

    try:
        (directory / mc_llm_setup.RUNTIME_MARKER).write_text(build.component_id,
                                                             encoding="utf-8")
        atomic_write_json(directory / PROVENANCE_MARKER, {
            "component_id": build.component_id,
            "family": build.family,
            "commit": build.commit,
            "pull_request": build.pull_request,
            "cuda": build.cuda,
            "source": source,
            "installed_at": time.time(),
        })
    except OSError:
        logger.warning("Model Chain: could not record which DFlash2 build was installed",
                       exc_info=True)


# --------------------------------------------------------------------------- #
# Proving it
# --------------------------------------------------------------------------- #


def advertises(server: Path) -> bool:
    """Whether ``server``'s help lists the speculative options at all.

    A necessary condition and emphatically not a sufficient one -- see the
    module docstring. It is asked first because it is free: a build that cannot
    even name the flags will fail the real test after a two-minute model load,
    and refusing it in twenty seconds with the same answer is kinder.
    """
    import mc_llm_accel

    try:
        finished = subprocess.run([str(server), "--help"], capture_output=True, text=True,
                                  timeout=_help_timeout(),
                                  creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except Exception:
        logger.debug("Model Chain: could not ask the DFlash2 runtime what it supports",
                     exc_info=True)
        return False
    text = f"{finished.stdout or ''}\n{finished.stderr or ''}"
    return (mc_llm_accel.SPEC_MODEL_FLAG in text and mc_llm_accel.SPEC_TYPE_FLAG in text
            and mc_llm_accel.SPEC_TYPE_DFLASH in text)


def _help_timeout() -> float:
    """The same ``--help`` budget the ordinary runtime probe uses."""
    import mc_llm_runtime

    return float(getattr(mc_llm_runtime, "_HELP_TIMEOUT", 20))


@dataclass(frozen=True)
class Verification:
    """What one verification run proved, and what it could not."""

    component_id: str
    text: bool = False
    vision: bool = False
    detail: str = ""
    vision_detail: str = ""

    @property
    def passed(self) -> bool:
        return self.text


def verify(component_id: str = "", identifier: str = "", on_status=None,
           include_vision: bool = True) -> Verification:
    """Load the real target and the real sidecar, and make them answer.

    ``identifier`` is a managed backbone whose bundle holds both -- the Qwen 3.8
    tiers are the entries that have one. Everything about this is deliberately
    the *real* thing: the pinned runtime, the installed weights, the installed
    draft, the publisher's own flags, and a question with one right answer.
    Nothing here reads help text and concludes anything.

    Text and vision are two runs, and the second is skipped without prejudice
    when there is no projector or when the first failed. The results are
    recorded separately, so an installation can sit at "text yes, vision no" --
    which is a real state of this pull request and is the state a forced
    DFlash2 image request has to be told about rather than crash into.

    Every server this starts is started under the workload lock, with whatever
    llama-server was running stopped first. That is the same rule a backbone
    switch keeps and for the same reason: this loads a 20 GB target and a
    3.86 GB draft, and doing it beside a resident model or an image generation
    is not a verification, it is an out-of-memory error in somebody else's
    work.
    """
    import mc_broker

    say = on_status or (lambda _text: None)

    component_id = component_id or installed_component()
    server = executable(component_id) if component_id else None
    if server is None:
        raise DFlashError("There is no DFlash2 runtime installed to verify. Install one in "
                          "Setup first; the ordinary runtime is unaffected.")

    bundle, model = _verifiable_bundle(identifier)
    speculator = model.accelerators.dflash2
    if speculator is None or bundle.draft is None:
        raise DFlashError(
            f"{model.label} has no draft model installed, so there is nothing to verify "
            f"DFlash2 against. Install the draft in Setup first."
        )

    say("Reading what this build advertises…")
    if not advertises(server):
        detail = ("This build's help text does not list the speculative draft options, so it "
                  "is not the DFlash2 branch.")
        record_capability(component_id, text=False, vision=False, detail=detail)
        return Verification(component_id, False, False, detail)

    with mc_broker.workload(mc_broker.FAMILY_LLM, "verifying the DFlash2 runtime",
                            timeout=WORKLOAD_TIMEOUT):
        _stop_whatever_is_running()

        say("Loading the backbone and its draft model…")
        passed, detail = _smoke(server, bundle, speculator, projector=None)
        record_capability(component_id, text=passed, detail=detail)
        if not passed:
            return Verification(component_id, False, False, detail)

        if not include_vision or bundle.mmproj is None:
            skipped = ("No vision projector is installed, so DFlash2 vision was not tested "
                       "and stays unavailable." if bundle.mmproj is None else
                       "DFlash2 vision was not tested on this run and stays unavailable.")
            record_capability(component_id, vision=False, vision_detail=skipped)
            return Verification(component_id, True, False, detail, skipped)

        say("Sending one image through the projector…")
        saw, vision_detail = _smoke(server, bundle, speculator, projector=bundle.mmproj)
        record_capability(component_id, vision=saw, vision_detail=vision_detail)
        return Verification(component_id, True, saw, detail, vision_detail)


WORKLOAD_TIMEOUT = 20.0
"""How long a verification waits for the GPU before saying something else has it.

The same twenty seconds a backbone switch waits, and for the same reason: a
user watching a button would rather read "Stable Diffusion is using the GPU"
now than watch it do nothing and then succeed.
"""

STOP_TIMEOUT = 30.0
"""How long to wait for a running llama-server to actually be gone.

``stop()`` returning is a statement about a handle rather than about a process:
on Windows a server can still be unmapping a 20 GB file for several seconds
after it has been asked to exit, and starting the verification into that is how
a good build fails its own test."""


def _stop_whatever_is_running() -> None:
    """Stop the managed llama-server, and observe that it is gone."""
    import mc_llm_runtime

    mc_llm_runtime.runtime.stop()
    deadline = time.monotonic() + STOP_TIMEOUT
    while time.monotonic() < deadline:
        if not mc_llm_runtime.runtime.running():
            return
        time.sleep(0.25)
    raise DFlashError(
        "The llama-server that was running has not exited, so there is no room to verify "
        "the DFlash2 runtime beside it. Press Unload in the model sheet and try again; "
        "nothing was changed."
    )


def _verifiable_bundle(identifier: str):
    """The installed bundle to verify against, and its catalogue entry.

    ``identifier`` empty means "whichever installed backbone has a draft", which
    is what the panel's button passes: there is at most one such family today
    and asking a user to name it would be asking them to know something the
    extension already knows.
    """
    import mc_llm_managed_models

    if identifier:
        model = mc_llm_managed_models.entry(identifier)
        bundle = mc_llm_managed_models.installed(model.identifier)
        if bundle is None:
            raise DFlashError(f"{model.label} is not downloaded, so there is nothing to "
                              f"verify DFlash2 against.")
        return bundle, model

    for model in mc_llm_managed_models.catalogue():
        if model.draft is None:
            continue
        bundle = mc_llm_managed_models.installed(model.identifier)
        if bundle is not None and bundle.drafts(model):
            return bundle, model
    raise DFlashError("No managed backbone with a DFlash2 draft model is installed. Download "
                      "one, add its draft, and verify again.")


def _smoke(server: Path, bundle, speculator, projector) -> tuple[bool, str]:
    """One real llama-server, one real question, one answer or one reason.

    Started outside :class:`mc_llm_runtime.Runtime` on purpose. This is a
    *test* of a program, not a placement of one: it must not adopt the warm
    server, must not be handed back to the next request, must not register a
    residency, and must not leave anything running afterwards -- and the
    simplest way to be certain of all four is not to involve the thing that
    does them.
    """
    import mc_llm_accel
    import mc_llm_runtime

    mc_llm_runtime._repair_launcher()
    paths = mc_llm_paths.app_paths()
    paths.logs.mkdir(parents=True, exist_ok=True)
    log_path = paths.logs / "llama-server-dflash2.log"
    written_before = log_path.stat().st_size if log_path.exists() else 0

    flags = mc_llm_accel.dflash2_flags(
        speculator, bundle.draft,
        supports=lambda flag: mc_llm_runtime.runtime_supports(flag, _probe_config(server)),
        accepts=lambda value: mc_llm_runtime.runtime_accepts(
            mc_llm_accel.SPEC_TYPE_FLAG, value, _probe_config(server)),
        flash_attention=_flash_attention(server))
    if not flags:
        return False, ("This build does not accept the speculative draft options, so DFlash2 "
                       "cannot be started with it.")

    from prompt_master.inference.llama_process import LlamaProcess

    process = LlamaProcess()
    device, index = _verification_device()
    try:
        mc_llm_runtime._arm_flags(flags)
        process.start(server, bundle.model, projector, index, device,
                      SMOKE_CONTEXT, log_path, gpu_layers="999",
                      cache_type_k="q8_0", cache_type_v="q8_0", jinja=True)
        process.wait_ready(SMOKE_TIMEOUT)
        answered = _ask(process, projector)
    except Exception as exc:
        said = mc_llm_runtime.read_failure(
            mc_llm_runtime._text_since(log_path, written_before))
        return False, said.text or f"{type(exc).__name__}: {exc}"
    finally:
        # Armed flags are consumed by the launch, but a start that never
        # reached it would leave them for whatever started next -- which on
        # this path is an ordinary server. See ``mc_llm_runtime._arm_flags``.
        mc_llm_runtime._arm_flags(())
        process.stop()

    if projector is None:
        if SMOKE_ANSWER.casefold() not in answered.casefold():
            return False, (f"The server started and answered {answered.strip()!r} rather than "
                           f"{SMOKE_ANSWER!r}, so this build loads the model without running "
                           f"it correctly.")
        return True, f"Answered {SMOKE_ANSWER!r} with the draft model resident."
    if not answered.strip():
        return False, ("The server started with the projector and returned nothing for an "
                       "image request.")
    return True, f"Answered {answered.strip()[:60]!r} for an image request."


SMOKE_CONTEXT = 2048
"""Context for a verification run. Small deliberately: this proves a mechanism,
and every token of context is VRAM that makes the proof harder to fit."""


def _probe_config(server: Path):
    """A configuration carrying nothing but the executable being probed.

    ``mc_llm_runtime.runtime_capabilities`` caches per executable, so asking it
    about this one costs one ``--help`` for the whole verification and nothing
    at all on the second call.
    """
    import mc_llm_runtime

    return mc_llm_runtime.Config(
        runtime=server, model=None, mmproj=None, gpu_index=0, device="CUDA0",
        gpu_layers="all", context_size=SMOKE_CONTEXT, context_mode="fixed",
        context_buffer_gb=0.0, kv_type_k="q8_0", kv_type_v="q8_0")


def _flash_attention(server: Path) -> tuple[str, ...]:
    """``--flash-attn``, in whichever of its two spellings this build has."""
    import mc_llm_runtime

    configuration = _probe_config(server)
    if not mc_llm_runtime.runtime_supports(mc_llm_runtime.FLASH_ATTENTION_FLAG, configuration):
        return ()
    if mc_llm_runtime._flash_attention_takes_a_value(configuration):
        return (mc_llm_runtime.FLASH_ATTENTION_FLAG, "on")
    return (mc_llm_runtime.FLASH_ATTENTION_FLAG,)


def _verification_device() -> tuple[str, int]:
    """Which card to verify on: the one the installation is configured for."""
    try:
        import mc_llm_runtime

        configuration = mc_llm_runtime.config()
        if configuration.uses_cuda_compute:
            return str(configuration.device), int(configuration.gpu_index)
    except Exception:
        logger.debug("Model Chain: could not read which card to verify DFlash2 on",
                     exc_info=True)
    return "CUDA0", 0


def _ask(process, projector) -> str:
    """One deterministic request against a server that has just come up."""
    from prompt_master.inference.llama_client import LlamaClient

    client = LlamaClient(f"http://127.0.0.1:{process.port}", process.api_key)
    if projector is None:
        messages = [{"role": "user", "content": SMOKE_PROMPT}]
        tokens = SMOKE_TOKENS
    else:
        messages = [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": _swatch()}},
            {"type": "text", "text": VISION_SMOKE_PROMPT},
        ]}]
        tokens = VISION_SMOKE_TOKENS
    return client.stream_chat(messages, max_tokens=tokens, seed=1,
                              on_text=lambda _chunk: None, temperature=0.0, top_p=1.0)


def _swatch() -> str:
    """A two-by-two PNG of one flat colour, as a data URL.

    Built here rather than checked in as a file, because a four-pixel image is
    smaller as the three lines that make it than as an asset somebody has to
    keep beside the code and remember why.
    """
    import base64
    import io

    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (2, 2), VISION_SMOKE_COLOUR).save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")
