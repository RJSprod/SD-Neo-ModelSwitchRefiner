"""The gallery images behind a LoRA's prompts: downloaded once into one flat
folder, found again by looking on disk, thumbnailed for the cards, and
fetched in bulk by a background job.

Carried in with the LoRA details (docs/26). The image host is a local HTTP
server here, so the download path is exercised for real.
"""

from __future__ import annotations

import http.server
import os
import socketserver
import threading
import time
from pathlib import Path

import pytest

import mc_lora_images as images
import mc_lora_jobs as jobs


class ImageServer:
    """A tiny local HTTP server: a JPEG for any path, a 404 for /missing, a
    WebP by content type alone for /noext."""

    BODY = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32

    def __enter__(self):
        body = self.BODY

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *_args):
                pass

            def do_GET(self):
                if self.path == "/missing":
                    self.send_response(404)
                    self.end_headers()
                    return
                content_type = "image/webp" if self.path == "/noext" else "image/jpeg"
                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self.server = socketserver.TCPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.base = f"http://127.0.0.1:{self.server.server_address[1]}"
        return self

    def __exit__(self, *_exc):
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def server():
    with ImageServer() as served:
        yield served


class TestDownloads:
    def test_an_image_downloads_once_and_reappears_as_new_when_deleted(self, tmp_path, server):
        url = f"{server.base}/cat.jpeg"

        assert images.find_local(url, "42", directory=tmp_path) is None

        first = images.download(url, "42", directory=tmp_path)
        assert first.ok and not first.skipped
        assert first.path.name == "civitai-42.jpeg", first.path.name

        second = images.download(url, "42", directory=tmp_path)
        assert second.ok and second.skipped, "an already-present image must not be re-fetched"

        # the whole point: presence is answered from disk, never from a manifest
        first.path.unlink()
        assert images.find_local(url, "42", directory=tmp_path) is None
        third = images.download(url, "42", directory=tmp_path)
        assert third.ok and not third.skipped

    def test_image_names_are_stable_without_a_civitai_id(self, tmp_path, server):
        url = f"{server.base}/noext"

        result = images.download(url, directory=tmp_path)
        assert result.ok
        assert result.path.name.startswith("url-") and result.path.suffix == ".webp"
        assert images.find_local(url, directory=tmp_path) == result.path
        assert images.stem_for(url) == images.stem_for(url), "the name must not drift between runs"

    def test_image_download_refuses_what_it_should(self, tmp_path, server):
        assert not images.download("", directory=tmp_path).ok
        assert "no image URL" in images.download("", directory=tmp_path).message

        refused = images.download("file:///etc/passwd", directory=tmp_path)
        assert not refused.ok and "non-HTTP" in refused.message

        missing = images.download(f"{server.base}/missing", "7", directory=tmp_path)
        assert not missing.ok and "404" in missing.message

        capped = images.download(f"{server.base}/big.jpeg", "8", directory=tmp_path, max_bytes=8)
        assert not capped.ok and "limit" in capped.message

        assert not list(tmp_path.glob("*.part")), "a failed download must leave no partial file"
        assert images.stats(tmp_path)["files"] == 0

    def test_purge_empties_the_image_folder(self, tmp_path, server):
        images.download(f"{server.base}/a.jpeg", "1", directory=tmp_path)
        images.download(f"{server.base}/b.jpeg", "2", directory=tmp_path)
        assert images.stats(tmp_path)["files"] == 2

        removed, message = images.purge(tmp_path)
        assert removed == 2 and "Deleted 2" in message
        assert images.stats(tmp_path)["files"] == 0

    def test_the_folder_is_images_under_the_data_root_unless_a_setting_says(self, lora_root, host, monkeypatch):
        assert images.image_dir() == lora_root / "images"
        assert (lora_root / "images").is_dir(), "made on demand"

        elsewhere = lora_root.parent / "pictures"
        monkeypatch.setattr(host.shared.opts, "model_chain_lora_image_dir", str(elsewhere), raising=False)
        assert images.image_dir() == elsewhere and elsewhere.is_dir()

    def test_an_image_folder_that_cannot_be_made_falls_back_to_the_default(self, lora_root, host, monkeypatch, caplog):
        import logging

        blocked = lora_root.parent / "file-not-folder"
        blocked.write_text("in the way", encoding="utf-8")
        monkeypatch.setattr(host.shared.opts, "model_chain_lora_image_dir", str(blocked / "images"), raising=False)
        with caplog.at_level(logging.INFO, logger="model_chain"):
            assert images.image_dir() == lora_root / "images"
        lines = [record.getMessage() for record in caplog.records]
        assert any("using the default" in line for line in lines), lines
        assert not any(str(blocked) in line for line in lines), "never a path in a log line"


class TestThumbnails:
    def test_thumbnails_are_made_once_and_embedded(self, tmp_path):
        Image = pytest.importorskip("PIL.Image")

        source = tmp_path / "civitai-1.png"
        Image.new("RGB", (900, 1200), (10, 20, 30)).save(source)
        cache = tmp_path / "thumbs"

        first = images.thumbnail(source, directory=cache)
        assert first is not None and first.suffix == ".webp"
        with Image.open(first) as thumb:
            assert max(thumb.size) <= images.THUMB_SIZE
        assert images.thumbnail(source, directory=cache) == first, "made once"
        assert images.thumbnail_data_uri(source, directory=cache).startswith("data:image/webp;base64,")

        # a replaced image gets a fresh thumbnail
        Image.new("RGB", (100, 100), (200, 0, 0)).save(source)
        stamp = time.time() + 5
        os.utime(source, (stamp, stamp))
        assert images.thumbnail(source, directory=cache) != first

        (tmp_path / "clip.mp4").write_bytes(b"\x00" * 16)
        assert images.thumbnail(tmp_path / "clip.mp4", directory=cache) is None
        assert images.thumbnail_data_uri(tmp_path / "missing.png", directory=cache) == ""

    def test_thumbnails_live_in_the_cache_under_the_data_root(self, lora_root):
        assert images.thumbnail_dir() == lora_root / "cache" / "thumbs"


class TestTheBulkJob:
    def test_bulk_image_download_reports_what_it_did(self, lora_root, host, server):
        job = jobs.submit_image_downloads(
            [
                (f"{server.base}/a.jpeg", "1"),
                (f"{server.base}/a.jpeg", "1"),   # duplicate -> already present
                (f"{server.base}/missing", "2"),  # 404 -> failed
                ("", "3"),                        # no url -> dropped before queueing
            ],
            label="Demo",
        )
        deadline = time.time() + 15
        while jobs.MANAGER.get(job.id).snapshot().active and time.time() < deadline:
            time.sleep(0.05)

        snapshot = jobs.MANAGER.get(job.id).snapshot()
        assert snapshot.state == jobs.DONE, snapshot.state
        assert "1 downloaded" in snapshot.summary
        assert "1 already present" in snapshot.summary
        assert "1 failed" in snapshot.summary
        assert (lora_root / "images" / "civitai-1.jpeg").is_file(), "into images/ under the data root"
        assert Path(snapshot.summary).name == snapshot.summary, "no path in what the job says"
