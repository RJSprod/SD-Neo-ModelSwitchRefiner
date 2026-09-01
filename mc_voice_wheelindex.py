"""Resolve one wheel from a publisher's package index, at install time.

Every other wheel this extension installs is named, sized and hashed in a
manifest committed to this repository. This module exists for the one closure
where that is not possible: PyTorch's CUDA builds are not on PyPI at all -- they
are published only on download.pytorch.org, which the machine that writes the
manifests cannot reach -- so the wheel cannot be pinned here in advance.

What this does instead is ask the publisher, in the same shape the model
downloads already use: read the publisher's index, find the one wheel that
matches this machine, and take the SHA-256 the publisher states for it. The
download is then checked against that digest exactly as a pinned one is.

That is a weaker claim than a pin and the difference is worth being exact
about. A pinned wheel is checked against a number this repository reviewed and
committed to; a resolved one is checked against a number its publisher states
today. Both refuse a corrupted or tampered download. Only the first refuses a
publisher who changed their mind. So this path is deliberately narrow: it is
reachable only for a closure the manifest marks as resolved, over HTTPS, from a
host the manifest names, and the digests that actually arrived are written into
the installed record so that a later change shows up as staleness rather than
as nothing.

The format is PEP 503's simple index -- a page of anchors whose href carries a
``#sha256=`` fragment -- which is what pip itself reads.
"""

from __future__ import annotations

import html.parser
import re
import urllib.parse

# A wheel filename, as PEP 427 defines it. The local version segment matters
# here: CUDA builds are "2.9.1+cu128", which arrives percent-encoded in an href
# and must survive the round trip or the version comparison silently fails.
_WHEEL = re.compile(
    r"^(?P<name>[A-Za-z0-9._]+)-(?P<version>[A-Za-z0-9._!+]+)"
    r"(?:-(?P<build>[0-9][A-Za-z0-9._]*))?"
    r"-(?P<python>[A-Za-z0-9._]+)-(?P<abi>[A-Za-z0-9._]+)-(?P<platform>[A-Za-z0-9._]+)"
    r"\.whl$")


class IndexError_(RuntimeError):
    """Something that must stop a resolve, said in a sentence a user can act on."""


def normalise(name: str) -> str:
    """PEP 503's name normalisation. ``Torch_Vision`` and ``torch-vision`` are one."""
    return re.sub(r"[-_.]+", "-", str(name or "")).casefold()


def _sortable(version: str):
    """A version as something orderable, without depending on ``packaging``.

    The isolated interpreter this runs in has no third-party packages yet -- it
    is being built -- so this cannot import ``packaging.version``. Numeric runs
    compare as numbers and everything else as text, which orders the only shapes
    PyTorch publishes: ``2.9.1``, ``2.10.0``, ``2.9.1+cu128``. A local segment
    sorts after the release it decorates, which is why it is split off first.
    """
    text = str(version or "")
    release, _, local = text.partition("+")
    key = []
    for chunk in re.findall(r"\d+|[A-Za-z]+", release):
        key.append((1, int(chunk), "") if chunk.isdigit() else (0, 0, chunk.casefold()))
    return (key, local.casefold())


class _Anchors(html.parser.HTMLParser):
    """Every href on the page, in order. Nothing else about the HTML matters."""

    def __init__(self):
        super().__init__()
        self.hrefs: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag.casefold() != "a":
            return
        for key, value in attrs:
            if key.casefold() == "href" and value:
                self.hrefs.append(value)


def candidates(page: str, base: str) -> list:
    """Every wheel on an index page, as (filename, absolute url, digest).

    Split from the choosing so that the parsing can be tested against a real
    page and the selection against a hand-written one.
    """
    parser = _Anchors()
    parser.feed(str(page or ""))
    found = []
    for href in parser.hrefs:
        absolute = urllib.parse.urljoin(base, href)
        split = urllib.parse.urlsplit(absolute)
        name = urllib.parse.unquote(split.path.rsplit("/", 1)[-1])
        if not name.endswith(".whl"):
            continue
        digest = ""
        for key, value in urllib.parse.parse_qsl(split.fragment):
            if key.casefold() == "sha256":
                digest = str(value or "").casefold()
        found.append((name, urllib.parse.urlunsplit(split._replace(fragment="")), digest))
    return found


def choose(page: str, base: str, package: str, version: str, tags) -> dict:
    """The one wheel on this page for this package, version and machine.

    ``tags`` is an ordered preference -- ``("cp313-cp313-win_amd64", …)`` -- and
    the first tag with a match wins, so a machine-specific wheel is taken over a
    universal one even when the publisher offers both.

    Every refusal below is a refusal to install *something*, which is the right
    outcome. The failure this guards against is not a missing wheel, which is
    obvious; it is quietly installing the wrong one -- a different Python, a
    different CUDA build, a name that merely starts the same -- which is a
    runtime that imports and then does not work on this machine.
    """
    wanted_name = normalise(package)
    wanted_version = str(version or "")
    rows = candidates(page, base)
    if not rows:
        raise IndexError_(
            f"The publisher's index for {package} lists no wheels at all. That is "
            f"usually a URL that reached a page rather than an index.")

    matched = {}
    for name, url, digest in rows:
        found = _WHEEL.match(name)
        if found is None or normalise(found.group("name")) != wanted_name:
            continue
        version = found.group("version")
        if wanted_version and version != wanted_version:
            continue
        tag = f"{found.group('python')}-{found.group('abi')}-{found.group('platform')}"
        # Newest wins per tag, rather than whichever the publisher happened to
        # list first. The manifest deliberately does not pin a version for this
        # path -- it cannot, since the machine that writes it cannot see the
        # index -- so "newest" has to be a decision made here and made the same
        # way twice, not an accident of page order.
        seen = matched.get(tag)
        if seen is None or _sortable(version) > _sortable(seen[0]):
            matched[tag] = (version, name, url, digest)

    for tag in tags:
        if tag not in matched:
            continue
        _, name, url, digest = matched[tag]
        if not url.startswith("https://"):
            raise IndexError_(f"The publisher offers {name} over something other than "
                              f"HTTPS. Nothing was downloaded.")
        if "/" in name or "\\" in name or name in (".", ".."):
            raise IndexError_(f"The publisher's index names a file ({name!r}) that would "
                              f"not stay in its own folder.")
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            # No digest means nothing to check the download against, and this
            # path has no pinned digest to fall back on. Refusing is the whole
            # point: an unverifiable wheel is worse than an absent one.
            raise IndexError_(
                f"The publisher's index gives no SHA-256 for {name}, so a download "
                f"could not be checked against anything. Nothing was installed.")
        return {"filename": name, "url": url, "sha256": digest, "tag": tag}

    if matched:
        raise IndexError_(
            f"The publisher has {package} {wanted_version or ''}".rstrip() +
            f" but not for this machine. It offers "
            f"{', '.join(sorted(matched)[:4])}; this machine needs one of "
            f"{', '.join(tags[:3])}.")
    raise IndexError_(
        f"The publisher's index has no {package} "
        f"{wanted_version or '(any version)'} at all.")


def platform_tags(python: str, system: str, machine: str) -> tuple:
    """The wheel tags this machine can run, most specific first.

    Derived from the manifest's own platform identity rather than from the
    running interpreter, because the installer builds a closure for the platform
    the manifest names and those must not be allowed to disagree.
    """
    digits = "".join(ch for ch in str(python or "") if ch.isdigit() or ch == ".")
    parts = digits.split(".")
    if len(parts) < 2 or not parts[0] or not parts[1]:
        raise IndexError_(f"{python!r} is not a Python version this can build tags from.")
    short = f"cp{parts[0]}{parts[1]}"
    system = str(system or "").casefold()
    machine = str(machine or "").casefold()
    if system == "windows":
        plat = "win_amd64" if machine in ("amd64", "x86_64") else "win32"
    elif system == "darwin":
        plat = "macosx_11_0_arm64" if machine in ("arm64", "aarch64") else "macosx_10_9_x86_64"
    else:
        plat = ("manylinux2014_aarch64" if machine in ("arm64", "aarch64")
                else "manylinux2014_x86_64")
    return (f"{short}-{short}-{plat}", f"{short}-abi3-{plat}", f"{short}-none-{plat}",
            f"py3-none-{plat}", "py3-none-any", "py2.py3-none-any")
