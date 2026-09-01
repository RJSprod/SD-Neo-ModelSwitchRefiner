"""Resolving one wheel from a publisher's index, and refusing the near misses.

This module exists for the single closure that cannot be pinned in advance --
PyTorch's CUDA builds, published only on a host the manifest-writing machine
cannot reach -- so the checks it performs are the only thing standing between a
user clicking one button and an arbitrary file being unpacked into an
interpreter that then runs on their machine.

The interesting cases are not "no wheel found", which is loud. They are the
quiet ones: a package whose name merely starts the same, a wheel for a
different Python that would import and then fail, an index entry with no digest
to check a download against.
"""

from __future__ import annotations

import pytest

import mc_voice_wheelindex as index

DIGEST = "a" * 64
OTHER = "b" * 64
BASE = "https://download.pytorch.org/whl/cu128/torch/"


def page(*rows: str) -> str:
    return "<!DOCTYPE html><html><body>" + "".join(rows) + "</body></html>"


def link(name: str, digest: str = DIGEST, host: str = "") -> str:
    import urllib.parse
    quoted = urllib.parse.quote(name)
    frag = f"#sha256={digest}" if digest else ""
    return f'<a href="{host}/whl/cu128/{quoted}{frag}">{name}</a><br>'


WINDOWS = index.platform_tags("3.13", "windows", "amd64")


class TestItFindsTheRightWheel:

    def test_the_machines_own_python_is_chosen_over_another(self):
        found = index.choose(
            page(link("torch-2.9.1+cu128-cp312-cp312-win_amd64.whl", OTHER),
                 link("torch-2.9.1+cu128-cp313-cp313-win_amd64.whl", DIGEST)),
            BASE, "torch", "2.9.1+cu128", WINDOWS)
        assert found["tag"] == "cp313-cp313-win_amd64"
        assert found["sha256"] == DIGEST

    def test_a_cuda_local_version_survives_percent_encoding(self):
        """``2.9.1+cu128`` arrives as ``2.9.1%2Bcu128``.

        If the unquoting is skipped the version never matches and the resolve
        fails with "no such version" while the wheel is sitting right there.
        """
        found = index.choose(
            page(link("torch-2.9.1+cu128-cp313-cp313-win_amd64.whl")),
            BASE, "torch", "2.9.1+cu128", WINDOWS)
        assert found["filename"] == "torch-2.9.1+cu128-cp313-cp313-win_amd64.whl"
        assert "%2B" in found["url"], "the URL keeps the publisher's own encoding"

    def test_a_machine_specific_wheel_wins_over_a_universal_one(self):
        found = index.choose(
            page(link("thing-1.0-py3-none-any.whl", OTHER),
                 link("thing-1.0-cp313-cp313-win_amd64.whl", DIGEST)),
            BASE, "thing", "1.0", WINDOWS)
        assert found["sha256"] == DIGEST

    def test_a_universal_wheel_is_taken_when_it_is_all_there_is(self):
        found = index.choose(page(link("thing-1.0-py3-none-any.whl")),
                             BASE, "thing", "1.0", WINDOWS)
        assert found["tag"] == "py3-none-any"

    def test_any_version_is_allowed_when_none_is_demanded(self):
        found = index.choose(page(link("thing-9.9-cp313-cp313-win_amd64.whl")),
                             BASE, "thing", "", WINDOWS)
        assert found["filename"].startswith("thing-9.9")

    def test_the_publishers_name_spelling_does_not_have_to_match_ours(self):
        """PEP 503: ``huggingface_hub`` and ``huggingface-hub`` are one name."""
        found = index.choose(page(link("huggingface_hub-1.0-py3-none-any.whl")),
                             BASE, "huggingface-hub", "1.0", WINDOWS)
        assert found["sha256"] == DIGEST


class TestItRefusesTheQuietNearMisses:

    def test_a_name_that_merely_starts_the_same_is_not_a_match(self):
        """``torchvision`` is not ``torch``, and a prefix match would install it."""
        with pytest.raises(index.IndexError_, match="no torch"):
            index.choose(page(link("torchvision-0.19-cp313-cp313-win_amd64.whl")),
                         BASE, "torch", "", WINDOWS)

    def test_a_wheel_with_no_digest_is_refused_rather_than_trusted(self):
        """There is no pinned digest on this path to fall back to.

        An unverifiable wheel is worse than an absent one: absent stops the
        install, unverifiable finishes it.
        """
        with pytest.raises(index.IndexError_, match="no SHA-256"):
            index.choose(page(link("torch-2.9.1-cp313-cp313-win_amd64.whl", digest="")),
                         BASE, "torch", "2.9.1", WINDOWS)

    def test_a_truncated_or_malformed_digest_is_refused(self):
        with pytest.raises(index.IndexError_, match="no SHA-256"):
            index.choose(page(link("torch-2.9.1-cp313-cp313-win_amd64.whl", "abc")),
                         BASE, "torch", "2.9.1", WINDOWS)
        with pytest.raises(index.IndexError_, match="no SHA-256"):
            index.choose(page(link("torch-2.9.1-cp313-cp313-win_amd64.whl", "z" * 64)),
                         BASE, "torch", "2.9.1", WINDOWS)

    def test_a_wheel_offered_over_plain_http_is_refused(self):
        with pytest.raises(index.IndexError_, match="HTTPS"):
            index.choose(
                page(link("torch-2.9.1-cp313-cp313-win_amd64.whl", host="http://evil.test")),
                BASE, "torch", "2.9.1", WINDOWS)

    def test_the_wrong_version_is_not_quietly_substituted(self):
        with pytest.raises(index.IndexError_, match="no torch"):
            index.choose(page(link("torch-2.8.0-cp313-cp313-win_amd64.whl")),
                         BASE, "torch", "2.9.1", WINDOWS)

    def test_a_wheel_for_another_machine_says_what_it_has_instead(self):
        """The error names both sides, because this one is usually a real
        situation -- a publisher who has not built for this Python yet -- and
        the user can only act on it if they are told which is missing."""
        with pytest.raises(index.IndexError_) as caught:
            index.choose(page(link("torch-2.9.1-cp311-cp311-win_amd64.whl"),
                              link("torch-2.9.1-cp312-cp312-win_amd64.whl")),
                         BASE, "torch", "2.9.1", WINDOWS)
        message = str(caught.value)
        assert "cp311-cp311-win_amd64" in message
        assert "cp313-cp313-win_amd64" in message

    def test_an_index_with_no_wheels_is_probably_the_wrong_url(self):
        with pytest.raises(index.IndexError_, match="lists no wheels"):
            index.choose(page('<a href="/about">About</a>'), BASE, "torch", "", WINDOWS)

    def test_a_filename_that_would_escape_its_folder_is_refused(self):
        page_ = ('<a href="https://download.pytorch.org/x/'
                 '..%2F..%2Ftorch-1.0-cp313-cp313-win_amd64.whl'
                 f'#sha256={DIGEST}">t</a>')
        with pytest.raises(index.IndexError_):
            index.choose(page_, BASE, "torch", "1.0", WINDOWS)


class TestTheTagsDescribeTheMachineTheManifestNames:

    def test_windows_on_amd64_at_cpython_313(self):
        tags = index.platform_tags("3.13", "windows", "amd64")
        assert tags[0] == "cp313-cp313-win_amd64"
        assert "py3-none-any" in tags

    def test_the_ordering_puts_the_most_specific_first(self):
        tags = index.platform_tags("3.13", "windows", "x86_64")
        assert tags.index("cp313-cp313-win_amd64") < tags.index("py3-none-any")

    def test_linux_and_macos_are_described_too(self):
        assert index.platform_tags("3.12", "linux", "x86_64")[0].endswith("manylinux2014_x86_64")
        assert index.platform_tags("3.12", "darwin", "arm64")[0].endswith("macosx_11_0_arm64")

    def test_a_python_version_it_cannot_parse_is_refused_not_guessed(self):
        with pytest.raises(index.IndexError_):
            index.platform_tags("", "windows", "amd64")
        with pytest.raises(index.IndexError_):
            index.platform_tags("three", "windows", "amd64")


class TestTheNewestVersionIsChosenDeliberately:
    """The manifest cannot pin a version on this path, so "newest" is a rule.

    It has to be the same answer twice and it has to be a version comparison
    rather than a text one, because the moment PyTorch published 2.10 a
    string sort would have started preferring 2.9.
    """

    def test_ten_sorts_after_nine_rather_than_before_it(self):
        found = index.choose(
            page(link("torch-2.9.1+cu128-cp313-cp313-win_amd64.whl", OTHER),
                 link("torch-2.10.0+cu128-cp313-cp313-win_amd64.whl", DIGEST)),
            BASE, "torch", "", WINDOWS)
        assert found["filename"].startswith("torch-2.10.0")
        assert found["sha256"] == DIGEST

    def test_page_order_does_not_decide_it(self):
        newest = link("torch-2.10.0-cp313-cp313-win_amd64.whl", DIGEST)
        older = link("torch-2.9.1-cp313-cp313-win_amd64.whl", OTHER)
        first = index.choose(page(newest, older), BASE, "torch", "", WINDOWS)
        second = index.choose(page(older, newest), BASE, "torch", "", WINDOWS)
        assert first["filename"] == second["filename"]
        assert first["sha256"] == DIGEST

    def test_a_local_cuda_segment_sorts_after_the_release_it_decorates(self):
        assert index._sortable("2.9.1+cu128") > index._sortable("2.9.1")

    def test_an_explicit_version_still_overrides_the_newest_rule(self):
        found = index.choose(
            page(link("torch-2.10.0-cp313-cp313-win_amd64.whl", OTHER),
                 link("torch-2.9.1-cp313-cp313-win_amd64.whl", DIGEST)),
            BASE, "torch", "2.9.1", WINDOWS)
        assert found["filename"].startswith("torch-2.9.1")


class TestLiftingAPackageOutOfASourceArchive:
    """Three of LavaSR's dependencies publish source and no wheel.

    This writes files onto disk from an archive fetched over the network, so
    what it refuses matters as much as what it extracts. A tar member may name
    any path it likes, including one that climbs out of the destination.
    """

    @staticmethod
    def _archive(tmp_path, entries, name="pkg.tar.gz"):
        import io, tarfile
        path = tmp_path / name
        with tarfile.open(path, "w:gz") as bundle:
            for member, blob in entries.items():
                info = tarfile.TarInfo(member)
                info.size = len(blob)
                bundle.addfile(info, io.BytesIO(blob))
        return path

    def test_only_the_named_package_directory_is_taken(self, tmp_path):
        """A GitHub tarball is a repository, not a distribution.

        encodec's sdist carries four megabytes of sample audio next to its
        package; a LavaSR checkout carries a README and a pyproject. None of
        that belongs in a speech runtime.
        """
        import mc_voice_models as models
        archive = self._archive(tmp_path, {
            "LavaSR-33ac0408/README.md": b"# LavaSR\n",
            "LavaSR-33ac0408/pyproject.toml": b"[project]\n",
            "LavaSR-33ac0408/test_48k.wav": b"R" * 4096,
            "LavaSR-33ac0408/LavaSR/__init__.py": b"",
            "LavaSR-33ac0408/LavaSR/model.py": b"class LavaEnhance2:\n    pass\n",
            "LavaSR-33ac0408/LavaSR/enhancer/enhancer.py": b"# bwe\n"})
        added = models.unpack_source_archive(archive, "LavaSR", tmp_path / "site")
        assert (tmp_path / "site" / "LavaSR" / "model.py").exists()
        assert (tmp_path / "site" / "LavaSR" / "enhancer" / "enhancer.py").exists()
        assert not (tmp_path / "site" / "README.md").exists()
        assert not (tmp_path / "site" / "test_48k.wav").exists()
        assert all(name.startswith("LavaSR/") for name in added)

    def test_a_member_that_climbs_out_of_the_destination_is_refused(self, tmp_path):
        import mc_voice_models as models
        archive = self._archive(tmp_path, {
            "pkg-1.0/vocos/__init__.py": b"",
            "pkg-1.0/vocos/../../../escape.py": b"import os\n"})
        with pytest.raises(models.VoiceError, match="would not stay in its own folder"):
            models.unpack_source_archive(archive, "vocos", tmp_path / "site")

    def test_a_symlink_member_is_refused_rather_than_followed(self, tmp_path):
        """The one member that can still point outside after the path check."""
        import io, tarfile
        import mc_voice_models as models
        path = tmp_path / "link.tar.gz"
        with tarfile.open(path, "w:gz") as bundle:
            good = tarfile.TarInfo("pkg-1.0/vocos/__init__.py")
            good.size = 0
            bundle.addfile(good, io.BytesIO(b""))
            link = tarfile.TarInfo("pkg-1.0/vocos/secrets.py")
            link.type = tarfile.SYMTYPE
            link.linkname = "/etc/passwd"
            bundle.addfile(link)
        with pytest.raises(models.VoiceError, match="contains a link"):
            models.unpack_source_archive(path, "vocos", tmp_path / "site")

    def test_an_archive_without_the_package_says_so(self, tmp_path):
        import mc_voice_models as models
        archive = self._archive(tmp_path, {"other-1.0/other/__init__.py": b""})
        with pytest.raises(models.VoiceError, match="does not contain a 'vocos' package"):
            models.unpack_source_archive(archive, "vocos", tmp_path / "site")

    def test_a_package_name_that_is_a_path_is_refused(self, tmp_path):
        import mc_voice_models as models
        archive = self._archive(tmp_path, {"pkg-1.0/vocos/__init__.py": b""})
        for bad in ("../vocos", "a/b", "", ".."):
            with pytest.raises(models.VoiceError):
                models.unpack_source_archive(archive, bad, tmp_path / "site")

    def test_a_corrupt_archive_names_itself_rather_than_tracebacking(self, tmp_path):
        import mc_voice_models as models
        path = tmp_path / "broken.tar.gz"
        path.write_bytes(b"this is not a tarball")
        with pytest.raises(models.VoiceError, match="could not be read"):
            models.unpack_source_archive(path, "vocos", tmp_path / "site")

    def test_a_package_nested_deeper_is_still_found(self, tmp_path):
        """An sdist wraps in a version folder, a checkout in a commit folder."""
        import mc_voice_models as models
        archive = self._archive(tmp_path, {
            "encodec-0.1.1/src/encodec/__init__.py": b"",
            "encodec-0.1.1/src/encodec/model.py": b"class EncodecModel:\n    pass\n"})
        models.unpack_source_archive(archive, "encodec", tmp_path / "site")
        assert (tmp_path / "site" / "encodec" / "model.py").exists()
