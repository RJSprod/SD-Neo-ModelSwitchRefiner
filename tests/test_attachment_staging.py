"""Staged pictures: what is refused, and what "ready" is allowed to mean.

Readiness is the point of this module. Before it, "has the upload finished" was
answered by a timer -- right on a laptop, wrong on a phone with a twelve
megapixel camera -- and a message could be sent with a picture the server had
never looked at. It is a state the decode either reached or did not.

Decoding is also the attack surface, which is why most of these tests are
refusals. A three-hundred-byte PNG can decode to a forty-gigapixel canvas; an
SVG can carry script; a "JPEG" can be an HTML document a browser would happily
render from this application's own origin. Every one of those is answered before
the pixels are read, and what is kept is a raster this process built rather than
the file that arrived.
"""

from __future__ import annotations

import io

import pytest
from PIL import Image

import mc_llm_attachment_staging as staging


@pytest.fixture(autouse=True)
def clean():
    staging.reset()
    yield
    staging.reset()


def encoded(kind="PNG", size=(24, 18), colour=(200, 30, 30)):
    buffer = io.BytesIO()
    Image.new("RGB", size, colour).save(buffer, format=kind)
    return buffer.getvalue()


class TestAcceptance:
    @pytest.mark.parametrize("kind", ["PNG", "JPEG", "WEBP"])
    def test_a_supported_picture_becomes_ready(self, kind):
        found = staging.stage(encoded(kind), f"holiday.{kind.lower()}")

        assert found.state == staging.READY
        assert found.ready is True
        assert (found.width, found.height) == (24, 18)

    def test_what_is_kept_is_a_picture_this_process_built(self):
        """Not the bytes that arrived. Nothing of the original container -- its
        metadata, its trailing bytes, its second frame -- travels any further."""
        found = staging.stage(encoded(), "x.png")

        assert isinstance(found.image, Image.Image)
        assert found.image.mode == "RGB"

    def test_the_name_is_a_label_rather_than_a_path(self):
        found = staging.stage(encoded(), "../../etc/passwd")

        assert "/" not in found.name
        assert found.name == "passwd"

    def test_a_name_with_control_characters_is_cleaned(self):
        found = staging.stage(encoded(), "holiday\u0000‮.png")

        assert "\u0000" not in found.name
        assert "‮" not in found.name

    def test_a_very_long_name_is_bounded(self):
        found = staging.stage(encoded(), "x" * 4096 + ".png")

        assert len(found.name) <= 120


class TestRefusals:
    def test_an_empty_upload_is_refused(self):
        found = staging.stage(b"", "nothing.png")

        assert found.state == staging.FAILED
        assert found.ready is False

    def test_a_file_that_is_not_a_picture_is_refused(self):
        found = staging.stage(b"not a picture at all, just words", "x.png")

        assert found.state == staging.FAILED

    @pytest.mark.parametrize("body", [b"<svg xmlns='http://www.w3.org/2000/svg'></svg>",
                                      b"   <html><script>alert(1)</script></html>",
                                      b"<?xml version='1.0'?><svg/>"])
    def test_anything_that_would_be_rendered_is_refused_by_shape(self, body):
        """Refused by what it starts with rather than by its extension: the
        danger is that a browser would render it, and a file that starts with a
        tag is a file a browser would render."""
        found = staging.stage(body, "picture.png")

        assert found.state == staging.FAILED
        assert "Vector" in found.reason or "not a PNG" in found.reason

    def test_an_upload_beyond_the_size_limit_is_refused_before_decoding(self):
        found = staging.stage(b"\x89PNG\r\n\x1a\n" + b"\0" * staging.MAX_ENCODED_BYTES,
                              "huge.png")

        assert found.state == staging.FAILED
        assert "20 MB" in found.reason

    def test_a_decompression_bomb_is_refused(self):
        """A picture's dimensions are in its header, so this is answered before
        the pixel data is touched."""
        buffer = io.BytesIO()
        Image.new("RGB", (9000, 9000), (0, 0, 0)).save(buffer, format="PNG")

        found = staging.stage(buffer.getvalue(), "bomb.png")

        assert found.state == staging.FAILED
        assert "megapixel" in found.reason

    def test_an_animation_is_refused_rather_than_flattened(self):
        """Flattening would send a frame nobody chose."""
        buffer = io.BytesIO()
        frames = [Image.new("RGB", (8, 8), (index * 20, 0, 0)) for index in range(3)]
        frames[0].save(buffer, format="WEBP", save_all=True, append_images=frames[1:])

        found = staging.stage(buffer.getvalue(), "moving.webp")

        assert found.state == staging.FAILED
        assert "Animated" in found.reason


class TestLifecycle:
    def test_a_staged_picture_is_found_by_its_token(self):
        found = staging.stage(encoded(), "x.png")

        assert staging.resolve(found.token) is found

    def test_a_token_nothing_was_staged_under_resolves_to_nothing(self):
        assert staging.resolve("not-a-token") is None
        assert staging.resolve("") is None

    def test_a_used_picture_releases_its_bytes_and_stays_findable(self):
        """Findable, so a retried command naming the token gets "already used"
        rather than "expired" -- but the image itself is let go, because the
        message now owns a copy."""
        found = staging.stage(encoded(), "x.png")

        staging.consume(found.token)

        assert found.image is None
        assert staging.resolve(found.token) is None

    def test_removing_a_picture_forgets_it(self):
        found = staging.stage(encoded(), "x.png")

        assert staging.discard(found.token) is True
        assert staging.resolve(found.token) is None

    def test_an_expired_picture_is_swept(self, monkeypatch):
        found = staging.stage(encoded(), "x.png")
        found.created -= staging.STAGING_TTL + 1

        assert staging.expire() == 1
        assert staging.resolve(found.token) is None

    def test_a_pinned_picture_survives_the_sweep(self):
        """C06. The sweep must not be able to delete the attachment of a
        message that is halfway through being sent."""
        found = staging.stage(encoded(), "x.png")
        found.created -= staging.STAGING_TTL + 1
        staging.pin(found.token, "op-1")

        assert staging.expire() == 0
        assert staging.resolve(found.token) is not None

    def test_a_scope_cannot_hold_unbounded_uploads(self):
        for _ in range(staging.MAX_PER_SCOPE + 5):
            staging.stage(encoded(), "x.png", scope="one")

        assert len(staging.pending()) <= staging.MAX_PER_SCOPE + 1

    def test_two_scopes_do_not_evict_each_other(self):
        mine = staging.stage(encoded(), "mine.png", scope="mine")
        for _ in range(staging.MAX_PER_SCOPE + 5):
            staging.stage(encoded(), "theirs.png", scope="theirs")

        assert staging.resolve(mine.token) is not None
