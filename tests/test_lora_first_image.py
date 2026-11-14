"""Each LoRA's first image: kept with its prompt, recorded as the row's cover,
and downloaded after a fetch.

Asked for: "it is NOT downloading the first image and detailed prompt of the
first image for a lora. Lets make it so that atleast the first image (if
available) gets downloaded." A fetch downloaded no image at all, and the first
image could be lost from the library before that: one picture reaches a scan
twice -- in the version's showcase list and from the gallery endpoint -- and
the first build kept the copy seen first, the showcase one, which can come
without its generation data; with no prompt the picture was not recorded at
all, address included (docs/26 §9).

Civitai is the real client with its one network method, ``_get_json``, stood
in by pages shaped like the API's; downloads are stood in by a writer that puts
a file where the real one would.
"""

from __future__ import annotations

import time

import pytest

import mc_lora_civitai as civitai
import mc_lora_details
import mc_lora_images
import mc_lora_jobs as jobs
import mc_lora_render
from mc_lora_library import Library, parse_row

GUID_A = "0b2a1f3e-9c4d-4e5f-8a6b-7c8d9e0f1a2b"
GUID_B = "1c3b2a4f-0d5e-4f6a-9b7c-8d9e0f1a2b3c"
GUID_C = "2d4c3b5a-1e6f-4a7b-8c8d-9e0f1a2b3c4d"


def url(guid: str, size: str = "width=450", name: str = "1.jpeg") -> str:
    return f"https://image.civitai.com/xG1nkqKTMzGDvpLrqFT7WA/{guid}/{size}/{name}"


class Pages:
    """Stands in for ``CivitaiClient._get_json``: the gallery endpoint answers
    with the items given, in one page."""

    def __init__(self, items):
        self.items = items
        self.calls = 0

    def __call__(self, _url, *, params=None, allow_404=False):
        self.calls += 1
        return {"items": self.items, "metadata": {}}


def client_with(items) -> civitai.CivitaiClient:
    client = civitai.CivitaiClient(request_delay=0.0)
    client._get_json = Pages(items)
    return client


def showcase(guid, *, image_id=None, meta=None, name="1.jpeg"):
    image = {"url": url(guid, "width=450", name), "nsfwLevel": 1, "width": 832, "height": 1216, "hasMeta": True}
    if image_id is not None:
        image["id"] = image_id
    if meta is not None:
        image["meta"] = meta
    return image


def gallery_item(guid, image_id, prompt, created, name="1.jpeg"):
    meta = {"prompt": prompt, "negativePrompt": "blurry", "steps": 8} if prompt else None
    return {"id": image_id, "url": url(guid, "original=true", name), "createdAt": created, "meta": meta}


# --------------------------------------------------------------------------- #
# The gallery: one record a picture, the cover first
# --------------------------------------------------------------------------- #


class TestTheCover:
    def test_the_showcase_copy_without_a_prompt_takes_the_gallery_copys_prompt(self):
        """The first build kept the showcase copy alone: without generation data
        it had no prompt, so the cover was left out of the prompts and lost."""
        client = client_with([
            gallery_item(GUID_B, 202, "a later prompt", "2026-01-02T00:00:00Z", "202.jpeg"),
            gallery_item(GUID_A, 101, "the cover's prompt", "2026-01-03T00:00:00Z", "101.jpeg"),
        ])
        prompts, cover, warning = client.gallery(7, embedded_images=[showcase(GUID_A, image_id=101, name="101.jpeg")],
                                                 max_images=0)

        assert warning is None
        assert cover is not None and cover.prompt == "the cover's prompt" and cover.image_id == "101"
        assert [entry.prompt for entry in prompts] == ["the cover's prompt", "a later prompt"], \
            "the cover first, though it is not the oldest"
        assert prompts[0].image_url == cover.image_url

    def test_generation_data_without_a_prompt_gives_way_to_a_copy_with_one(self):
        """A showcase copy whose generation data is there but holds no prompt
        (a size, a seed) is not empty, so filling the empty fields alone would
        keep it; the copy that has the prompt brings its generation data."""
        client = client_with([gallery_item(GUID_A, 101, "the cover's prompt", "2026-01-03T00:00:00Z")])
        prompts, cover, _ = client.gallery(
            7, embedded_images=[showcase(GUID_A, image_id=101, meta={"Size": "832x1216", "seed": 4})],
            max_images=0)

        assert cover is not None and cover.prompt == "the cover's prompt"
        assert [entry.prompt for entry in prompts] == ["the cover's prompt"]

    def test_copies_without_an_id_are_one_picture_by_the_uuid_in_their_address(self):
        client = client_with([gallery_item(GUID_A, 101, "the cover's prompt", "2026-01-03T00:00:00Z")])
        prompts, cover, _ = client.gallery(7, embedded_images=[showcase(GUID_A)], max_images=0)

        assert cover is not None and cover.prompt == "the cover's prompt" and cover.image_id == "101"
        assert len(prompts) == 1, "one picture, not two"

    def test_a_cover_with_no_prompt_anywhere_is_still_the_cover(self):
        client = client_with([gallery_item(GUID_B, 202, "another picture's prompt", "2026-01-02T00:00:00Z")])
        prompts, cover, _ = client.gallery(
            7, embedded_images=[showcase(GUID_A, image_id=101), showcase(GUID_C, image_id=303)], max_images=0)

        assert cover is not None and cover.prompt == "" and cover.image_id == "101"
        assert GUID_A in cover.image_url
        assert [entry.prompt for entry in prompts] == ["another picture's prompt"]

    def test_with_no_showcase_the_oldest_picture_with_an_address_is_the_cover(self):
        client = client_with([
            gallery_item(GUID_B, 202, "newer", "2026-01-05T00:00:00Z"),
            gallery_item(GUID_A, 101, "", "2026-01-01T00:00:00Z"),
        ])
        prompts, cover, _ = client.gallery(7, embedded_images=[], max_images=0)

        assert cover is not None and cover.image_id == "101" and cover.prompt == ""
        assert [entry.prompt for entry in prompts] == ["newer"]

    def test_the_rest_keep_their_order_oldest_first(self):
        client = client_with([
            gallery_item(GUID_C, 303, "third", "2026-01-09T00:00:00Z"),
            gallery_item(GUID_B, 202, "second", "2026-01-04T00:00:00Z"),
            gallery_item(GUID_A, 101, "cover", "2026-01-07T00:00:00Z"),
        ])
        prompts, _, _ = client.gallery(7, embedded_images=[showcase(GUID_A, image_id=101)], max_images=0)

        assert [entry.prompt for entry in prompts] == ["cover", "second", "third"]

    def test_a_second_ask_answers_the_cover_from_the_cache(self):
        pages = Pages([gallery_item(GUID_A, 101, "cover", "2026-01-01T00:00:00Z")])
        client = civitai.CivitaiClient(request_delay=0.0)
        client._get_json = pages
        first = client.gallery(7, embedded_images=[showcase(GUID_A, image_id=101)], max_images=0)
        second = client.gallery(7, embedded_images=[showcase(GUID_A, image_id=101)], max_images=0)

        assert pages.calls == 1
        assert second[1] == first[1] and second[0] == first[0]
        assert client.gallery_prompts(7, embedded_images=[], max_images=0) == (first[0], None)


# --------------------------------------------------------------------------- #
# The row and the library
# --------------------------------------------------------------------------- #


class TestTheRow:
    def test_the_row_records_the_cover_and_the_library_reads_it(self):
        resolution = civitai.Resolution(
            "ok", model_name="m", prompts=(civitai.PromptImage("p2", url(GUID_B), "202"),),
            cover=civitai.PromptImage("", url(GUID_A), "101"))
        row = civitai.row_for("x.safetensors", "a" * 64, "", resolution)
        assert row["cover_image_url"] == url(GUID_A) and row["cover_image_id"] == "101"

        record = parse_row(row, source="s.csv", row_number=2)
        assert record.cover is not None and record.first_image == record.cover
        assert record.first_image.text == "" and record.first_image.image_id == "101"
        assert "cover_image_url" not in record.extras, "a column the library reads is not an extra field"

    def test_a_cover_that_is_a_prompts_picture_carries_its_prompt(self):
        resolution = civitai.Resolution(
            "ok", model_name="m", prompts=(civitai.PromptImage("the prompt", url(GUID_A), "101"),),
            cover=civitai.PromptImage("the prompt", url(GUID_A), "101"))
        record = parse_row(civitai.row_for("x.safetensors", "a" * 64, "", resolution), source="s", row_number=2)
        assert record.first_image.text == "the prompt"

    def test_a_row_from_before_covers_has_its_first_prompts_picture(self):
        record = parse_row({"safetensor_file": "x.safetensors", "civitai_model_name": "m",
                            "positive_prompt_1": "p1", "positive_prompt_2": "p2",
                            "prompt_image_url_2": url(GUID_B), "prompt_image_id_2": "202"},
                           source="s", row_number=2)
        assert record.cover is None
        assert record.first_image.image_id == "202" and record.first_image.text == "p2"

    def test_the_csv_written_by_a_scan_has_the_cover(self, tmp_path, lora_root, monkeypatch):
        import csv
        import hashlib
        import struct
        import json

        loras = tmp_path / "loras"
        loras.mkdir()
        header = json.dumps({"__metadata__": {}}).encode()
        path = loras / "one.safetensors"
        path.write_bytes(struct.pack("<Q", len(header)) + header + b"x")
        sha = hashlib.sha256(path.read_bytes()).hexdigest()

        class Client:
            def __init__(self, **_kwargs):
                pass

            def close(self):
                pass

            def version_by_hash(self, _sha):
                return {"id": 7, "modelId": 8, "name": "v", "files": [{"hashes": {"SHA256": sha}}],
                        "images": [showcase(GUID_A, image_id=101)], "model": {"name": "M", "type": "LORA"}}

            def model_info(self, version, *, with_description=False):
                return version["model"]

            def gallery(self, _version_id, *, embedded_images, max_images):
                return (), civitai.PromptImage("", embedded_images[0]["url"], "101"), None

        monkeypatch.setattr(civitai, "CivitaiClient", Client)
        out = tmp_path / "out.csv"
        civitai.scan_folder(loras, out, fetch_descriptions=False, log=lambda _m: None)
        with out.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        assert rows[0]["cover_image_url"] == url(GUID_A) and rows[0]["cover_image_id"] == "101"


# --------------------------------------------------------------------------- #
# Downloading the first images
# --------------------------------------------------------------------------- #


@pytest.fixture
def downloads(monkeypatch, lora_root):
    """A download writes a small file where the real one would."""
    calls = []

    def fake(link, image_id="", *, directory=None, cancel=None, **_kwargs):
        calls.append((link, image_id))
        folder = directory or mc_lora_images.image_dir()
        folder.mkdir(parents=True, exist_ok=True)
        target = folder / f"{mc_lora_images.stem_for(link, image_id)}.jpeg"
        target.write_bytes(b"\xff\xd8")
        return mc_lora_images.DownloadResult(True, target, "saved")

    monkeypatch.setattr(mc_lora_images, "download", fake)
    return calls


def wait(job, seconds=15):
    deadline = time.time() + seconds
    while jobs.MANAGER.get(job.id).snapshot().active and time.time() < deadline:
        time.sleep(0.02)
    return jobs.MANAGER.get(job.id).snapshot()


def record(name, *, cover=None, prompts=()):
    row = {"safetensor_file": f"{name}.safetensors", "civitai_model_name": name}
    if cover:
        row["cover_image_url"], row["cover_image_id"] = cover
    for index, (text, link, image_id) in enumerate(prompts, start=1):
        row[f"positive_prompt_{index}"] = text
        row[f"prompt_image_url_{index}"] = link
        row[f"prompt_image_id_{index}"] = image_id
    return parse_row(row, source="s.csv", row_number=2)


class TestDownloading:
    def test_each_lora_gets_its_first_image_once(self, downloads):
        records = [
            record("a", cover=(url(GUID_A), "101"), prompts=[("p", url(GUID_B), "202")]),
            record("b", prompts=[("p", url(GUID_C), "303")]),
            record("c"),
            record("a-again", cover=(url(GUID_A), "101")),
        ]
        job = jobs.Job("t", "t", None)
        saved, present, failed = jobs.download_first_images(job, records)

        assert downloads == [(url(GUID_A), "101"), (url(GUID_C), "303")]
        assert (saved, present, failed) == (2, 0, 0)
        assert jobs.download_first_images(job, records) == (0, 2, 0), "one already there is not fetched again"
        assert len(downloads) == 2

    def test_the_library_job_downloads_every_first_image(self, downloads):
        import mc_lora_library

        library = mc_lora_library.get_library()
        (library.directory / "l.csv").write_text(
            "safetensor_file,civitai_model_name,cover_image_url,cover_image_id\n"
            f"a.safetensors,A,{url(GUID_A)},101\n", encoding="utf-8")
        library.reload()
        snapshot = wait(jobs.submit_first_images())

        assert snapshot.state == jobs.DONE, snapshot
        assert downloads == [(url(GUID_A), "101")]
        assert "1 downloaded" in snapshot.summary


class TestTheFetchJob:
    @pytest.fixture
    def scan(self, tmp_path, lora_root, monkeypatch, downloads):
        """A fetch of a folder of one new LoRA into a CSV that already holds an
        earlier row, the way a refresh finds the library."""
        import hashlib
        import json
        import struct

        import mc_lora_library

        loras = tmp_path / "loras"
        loras.mkdir()
        header = json.dumps({"__metadata__": {}}).encode()
        path = loras / "new.safetensors"
        path.write_bytes(struct.pack("<Q", len(header)) + header + b"n")
        sha = hashlib.sha256(path.read_bytes()).hexdigest()

        class Client:
            def __init__(self, **_kwargs):
                pass

            def close(self):
                pass

            def version_by_hash(self, _sha):
                return {"id": 7, "modelId": 8, "name": "v", "files": [{"hashes": {"SHA256": sha}}],
                        "model": {"name": "New", "type": "LORA"}}

            def model_info(self, version, *, with_description=False):
                return version["model"]

            def gallery(self, _version_id, *, embedded_images, max_images):
                return (), civitai.PromptImage("", url(GUID_A), "101"), None

        monkeypatch.setattr(civitai, "CivitaiClient", Client)
        library = mc_lora_library.get_library()
        (library.directory / "civitai-loras.csv").write_text(
            "safetensor_file,civitai_model_name,positive_prompt_1,prompt_image_url_1,prompt_image_id_1\n"
            f"old.safetensors,Old,a prompt,{url(GUID_B)},202\n", encoding="utf-8")
        library.reload()

        def run():
            return wait(jobs.submit_scan(loras, incremental=False, fetch_descriptions=False, request_delay=0.0))

        return run

    def test_a_fetch_downloads_the_first_image_of_every_row_it_wrote(self, scan, downloads):
        snapshot = scan()

        assert snapshot.state == jobs.DONE, snapshot
        assert sorted(downloads) == sorted([(url(GUID_A), "101"), (url(GUID_B), "202")]), \
            "the new LoRA's cover and the earlier row's first picture"
        assert "first images: 2 downloaded" in snapshot.summary

    def test_the_setting_off_downloads_nothing(self, scan, downloads):
        from modules import shared

        shared.opts.model_chain_lora_download_first_image = False
        try:
            snapshot = scan()
        finally:
            del shared.opts.model_chain_lora_download_first_image
        assert snapshot.state == jobs.DONE
        assert downloads == []
        assert "first images" not in snapshot.summary


# --------------------------------------------------------------------------- #
# The dialog
# --------------------------------------------------------------------------- #


class TestTheDialog:
    def test_a_first_image_with_no_prompt_has_a_block_of_its_own(self, lora_root):
        first = (url(GUID_A), "101")
        html = mc_lora_details._first_image(first, [(url(GUID_B), "202")])
        assert "First image" in html and "Not downloaded yet" in html

        folder = mc_lora_images.image_dir()
        from PIL import Image

        Image.new("RGB", (8, 12), (200, 40, 40)).save(folder / f"{mc_lora_images.stem_for(*first)}.png")
        shown = mc_lora_details._first_image(first, [])
        assert '<img src="data:image/webp;base64,' in shown and "mc-lora-first-image" in shown

    def test_none_where_the_gallery_shows_it_first(self, lora_root):
        first = (url(GUID_A), "101")
        assert mc_lora_details._first_image(first, [first, (url(GUID_B), "202")]) == ""
        assert mc_lora_details._first_image(None, []) == ""

    def test_the_viewer_takes_the_first_image_for_its_picture(self, lora_root):
        import mc_lora_viewer
        from PIL import Image

        first = (url(GUID_A), "101")
        other = (url(GUID_B), "202")
        folder = mc_lora_images.image_dir()
        Image.new("RGB", (8, 12), (0, 0, 0)).save(folder / f"{mc_lora_images.stem_for(*first)}.png")
        Image.new("RGB", (8, 12), (255, 255, 255)).save(folder / f"{mc_lora_images.stem_for(*other)}.png")
        picture = mc_lora_viewer.fallback_picture({"first_image": first, "prompt_images": [other]})
        assert picture == mc_lora_images.thumbnail_data_uri(folder / f"{mc_lora_images.stem_for(*first)}.png",
                                                             size=mc_lora_viewer.PICTURE_SIZE)

    def test_the_block_renders_with_the_host_colours_only(self):
        html = mc_lora_render.first_image_html(thumb="", saved=False, video=False, url=url(GUID_A))
        assert "image.civitai.com" in html and "style=" not in html
