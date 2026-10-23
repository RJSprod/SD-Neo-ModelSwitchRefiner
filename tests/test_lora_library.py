"""The CSV library behind the LoRA dialog: hashing, how a row is read, how a
LoRA is matched to one, and the text a description becomes.

Carried in with the LoRA details (docs/26). The library is the standalone
extension's: a folder of CSV files, each row a LoRA with its SHA256, its
Civitai facts, its trigger words, its prompts and the gallery images behind
them, matched to a card by hash first and by path or name when no row carries
the hash. Nothing here needs Gradio or the host beyond ``lora_root``, which
keeps every file under ``tmp_path``.
"""

from __future__ import annotations

import csv
import hashlib
import html as html_module
import json
import re
import struct
from pathlib import Path

import mc_lora_descriptions as descriptions
import mc_lora_hashing as hashing
import mc_lora_layouts as layouts
import mc_lora_render as render
from mc_lora_library import Library, parse_row


def make_safetensors(path: Path, payload: bytes, metadata: dict | None = None) -> str:
    """Write a minimal valid safetensors file; returns its full-file SHA256."""
    header = json.dumps({"__metadata__": metadata or {}}).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(struct.pack("<Q", len(header)) + header + payload)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def library_with_rows(directory: Path, rows: list[dict], columns: list[str], name="lib.csv") -> Library:
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / name).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    library = Library(directory)
    library.reload()
    return library


# --------------------------------------------------------------------- hashing


class TestHashing:
    def test_hashes_match_reference_digests(self, tmp_path):
        path = tmp_path / "a.safetensors"
        expected = make_safetensors(path, b"PAYLOAD", {"k": "v"})
        cache = hashing.HashCache(tmp_path / "cache.json")

        assert hashing.sha256_file(path, cache=cache) == expected
        assert hashing.addnet_sha256_file(path, cache=cache) == hashlib.sha256(b"PAYLOAD").hexdigest()

        # second read comes from the cache, and survives a save/load round trip
        assert cache.peek(path, "sha256") == expected
        cache.save()
        reloaded = hashing.HashCache(tmp_path / "cache.json")
        assert reloaded.peek(path, "sha256") == expected

        # touching the file invalidates the entry
        path.write_bytes(path.read_bytes() + b"more")
        assert reloaded.peek(path, "sha256") is None

    def test_hashing_a_missing_file_returns_none(self, tmp_path):
        cache = hashing.HashCache(tmp_path / "cache.json")
        assert hashing.sha256_file(tmp_path / "nope.safetensors", cache=cache) is None

    def test_the_cache_lives_under_the_data_root_and_is_made_when_first_written(self, lora_root):
        # Importing the module makes no folder; the path is resolved when asked.
        assert hashing.CACHE.path == lora_root / "cache" / "hashes.json"
        assert not (lora_root / "cache").exists()
        assert hashing.known_hash(lora_root / "nothing.safetensors") is None
        hashing.flush()
        assert not (lora_root / "cache").exists(), "nothing to write, nothing written"


# --------------------------------------------------------------------- parsing


class TestReadingARow:
    def test_parse_row_recognises_aliased_columns(self):
        record = parse_row(
            {
                "File Name": "sub/My Lora.safetensors",
                "SHA-256": "AB" * 32,
                "Model Name": "Cool Style",
                "Trained Words": "one, two; three",
                "prompt_2": "second",
                "prompt_1": "first",
                "image_url_1": "https://img/one.jpeg",
                "Notes": "kept as an extra",
                "Blank": "",
            },
            source="x.csv",
            row_number=2,
        )
        assert record.sha256 == "ab" * 32
        assert record.stem == "my lora"
        assert record.rel_path == "sub/my lora.safetensors"
        assert record.model_name == "Cool Style"
        assert record.trigger_words == ("one", "two", "three")
        assert record.prompts == ("first", "second")          # ordered by column index
        assert record.entries[0].image_url == "https://img/one.jpeg"
        assert record.entries[1].image_url == ""
        assert record.extras == {"Notes": "kept as an extra"}  # unknown columns survive
        assert "Blank" not in record.extras

    def test_parse_row_ignores_a_row_with_nothing_in_it(self):
        assert parse_row({"a": "", "b": None}, source="x.csv", row_number=2) is None

    def test_prompt_columns_sort_numerically_not_lexically(self):
        row = {f"positive_prompt_{i}": f"p{i}" for i in (1, 2, 10, 11)}
        record = parse_row(row, source="x.csv", row_number=2)
        assert record.prompts == ("p1", "p2", "p10", "p11")

    def test_an_empty_prompt_cell_does_not_shift_images_onto_the_wrong_prompt(self):
        record = parse_row(
            {
                "positive_prompt_1": "first",
                "prompt_image_url_1": "https://img/1.jpeg",
                "positive_prompt_2": "",                       # dropped
                "prompt_image_url_2": "https://img/2.jpeg",    # must go with it
                "positive_prompt_3": "third",
                "prompt_image_url_3": "https://img/3.jpeg",
                "prompt_image_id_3": "333",
            },
            source="x.csv",
            row_number=2,
        )
        assert record.prompts == ("first", "third")
        assert [entry.image_url for entry in record.entries] == ["https://img/1.jpeg", "https://img/3.jpeg"]
        assert record.entries[1].image_id == "333"

    def test_description_columns_are_read_and_not_repeated_as_extras(self):
        record = parse_row(
            {
                "sha256": "cd" * 32,
                "model_description": "About the model",
                "version_description": "About the version",
                "Notes": "kept",
            },
            source="x.csv",
            row_number=2,
        )
        assert record.description == "About the model"
        assert record.version_description == "About the version"
        assert record.has_description
        assert record.extras == {"Notes": "kept"}

        # an unrelated tool's column naming works just as well
        aliased = parse_row({"sha256": "ef" * 32, "Description": "text"}, source="y.csv", row_number=2)
        assert aliased.description == "text" and aliased.extras == {}

        assert not parse_row({"sha256": "ab" * 32, "civitai_model_name": "x"}, source="z.csv", row_number=2).has_description

    def test_the_record_carries_the_facts_the_header_shows(self):
        record = parse_row(dict(MOSS_ROW), source="civitai-moss.csv", row_number=2)
        assert (record.version_name, record.base_model, record.creator) == ("v1.0", "MOSS Image", "kohaku")
        assert (record.model_id, record.version_id, record.status, record.note) == ("11", "22", "ok", "fine")
        assert record.resolved and record.title == "Moss light"
        # read for the header, and still listed with the row's other fields
        assert record.extras["civitai_version_name"] == "v1.0" and record.extras["base_model"] == "MOSS Image"

        unresolved = parse_row(
            {"safetensor_file": "krea2\\funnyFaces_v1.safetensors", "sha256": "cd" * 32, "status": "not_found"},
            source="civitai-krea2.csv",
            row_number=3,
        )
        assert not unresolved.resolved
        assert unresolved.title == "funnyFaces_v1", "the file's own spelling, not the matching key"


# -------------------------------------------------------------------- matching


class TestMatching:
    def test_lookup_prefers_sha256_then_falls_back(self, tmp_path):
        sha = "cd" * 32
        library = library_with_rows(
            tmp_path,
            [
                {"safetensor_file": "sub/Hashed.safetensors", "sha256": sha, "civitai_model_name": "Hashed"},
                {"safetensor_file": "sub/Named.safetensors", "sha256": "", "civitai_model_name": "Named"},
            ],
            ["safetensor_file", "sha256", "civitai_model_name"],
        )

        records, how = library.lookup(sha256=sha.upper())
        assert how == "SHA256" and records[0].model_name == "Hashed"

        # a 12-char prefix of a full digest also resolves
        records, how = library.lookup(short_hashes=[sha[:12]])
        assert how == "short hash" and records[0].model_name == "Hashed"

        # backslash paths from a Windows CSV still match
        records, how = library.lookup(rel_paths=["sub\\Named.safetensors"])
        assert how == "file path" and records[0].model_name == "Named"

        # bare file name, and then stem-only, as last resorts
        records, how = library.lookup(filename="/anywhere/Named.safetensors")
        assert how == "file name" and records[0].model_name == "Named"

        assert library.lookup(sha256="00" * 32, filename="absent.safetensors") == ([], "")

    def test_matches_are_pooled_across_csv_files(self, tmp_path):
        sha = "ef" * 32
        library_with_rows(
            tmp_path,
            [{"safetensor_file": "a.safetensors", "sha256": sha, "positive_prompt_1": "from one"}],
            ["safetensor_file", "sha256", "positive_prompt_1"],
            name="one.csv",
        )
        library = library_with_rows(
            tmp_path,
            [{"safetensor_file": "a.safetensors", "sha256": sha, "positive_prompt_1": "from two"}],
            ["safetensor_file", "sha256", "positive_prompt_1"],
            name="two.csv",
        )
        records, how = library.lookup(sha256=sha)
        assert how == "SHA256" and len(records) == 2
        assert {record.source for record in records} == {"one.csv", "two.csv"}

    def test_a_broken_csv_does_not_take_the_library_down(self, tmp_path):
        (tmp_path / "empty.csv").write_text("", encoding="utf-8")
        (tmp_path / "binary.csv").write_bytes(b"\xff\xfe\x00\x01\x00")
        library = library_with_rows(
            tmp_path,
            [{"safetensor_file": "good.safetensors", "civitai_model_name": "Good"}],
            ["safetensor_file", "civitai_model_name"],
        )
        assert library.lookup(filename="good.safetensors")[0][0].model_name == "Good"

    def test_a_library_whose_folder_is_missing_is_empty_not_broken(self, tmp_path):
        library = Library(tmp_path / "not-there")
        assert library.reload() == []
        assert library.csv_files() == []
        assert library.lookup(filename="x.safetensors") == ([], "")

    def test_remove_csv_refuses_to_escape_the_library_directory(self, tmp_path):
        library = Library(tmp_path)
        library.reload()
        ok, message = library.remove_csv("../../etc/passwd")
        assert not ok and "outside the library directory" in message

    def test_the_shared_library_follows_the_setting(self, lora_root, host, monkeypatch):
        from mc_lora_library import get_library

        library = get_library()
        assert library.directory == lora_root / "library", "the default: library/ under the data root"
        assert (lora_root / "library").is_dir()

        elsewhere = lora_root.parent / "elsewhere"
        monkeypatch.setattr(host.shared.opts, "model_chain_lora_library_dir", str(elsewhere), raising=False)
        assert get_library() is library and library.directory == elsewhere
        assert elsewhere.is_dir(), "made on demand, like the default"


# --------------------------------------------------------------- descriptions


class TestDescriptions:
    def test_html_descriptions_become_readable_text(self):
        text = descriptions.html_to_text(
            "<h2>My LoRA</h2><p>Trained on <b>500</b> images.<br>Use <code>trigger</code>.</p>"
            "<ul><li>one</li><li>two <a href=\"/models/5\">the page</a></li></ul>"
            "<script>steal()</script><img src=\"https://img.example/x.png\">"
        )
        assert text == (
            "My LoRA\n\n"
            "Trained on 500 images.\n"
            "Use trigger.\n\n"
            "• one\n"
            "• two the page (https://civitai.com/models/5)"
        ), text
        assert "<" not in text, "no markup may survive the conversion"

    def test_description_conversion_keeps_lists_and_drops_nothing_else(self):
        assert descriptions.html_to_text("<ol><li>first</li><li>second<ul><li>nested</li></ul></li><li>third</li></ol>") == (
            "1. first\n2. second\n  • nested\n3. third"
        )
        # plain text passes through with its paragraphs and entities intact
        assert descriptions.html_to_text("Plain.\n\nSecond &amp; last.") == "Plain.\n\nSecond & last."
        assert descriptions.html_to_text(None) == "" and descriptions.html_to_text("   ") == ""
        # a link whose text is already the URL is not spelled out twice
        assert descriptions.html_to_text('<a href="https://x.example/y">https://x.example/y</a>') == "https://x.example/y"

    def test_a_long_description_is_capped_visibly(self):
        capped = descriptions.html_to_text("<p>" + "word " * 5000 + "</p>", limit=100)
        assert len(capped) == 100 and capped.endswith("…")

    def test_the_extractor_keeps_its_state_out_of_htmlparsers_way(self):
        # Python 3.13 gave HTMLParser a ``_pending`` buffer; an extractor that
        # also used the name broke close(), and every description fell back to
        # stripping.
        from html.parser import HTMLParser

        own = set(vars(descriptions._TextExtractor())) - set(vars(HTMLParser()))
        assert own and all(name.startswith("_text_") for name in own), own
        assert descriptions.html_to_text("<p>one</p><ul><li>two</li></ul>") == "one\n\n• two"

    def test_the_description_fallback_never_leaks_a_script(self, monkeypatch):
        def broken(self, data):
            raise ValueError("unparseable")

        monkeypatch.setattr(descriptions._TextExtractor, "feed", broken)
        text = descriptions.html_to_text("<p>Kept</p><script>steal()</script><style>p{}</style><b>too</b>")
        assert "Kept" in text and "too" in text
        assert "steal" not in text and "p{}" not in text, text

    def test_rendered_descriptions_escape_text_and_link_urls(self):
        html = render.rich_text("A <script>alert(1)</script> line\nsecond\n\nSee https://civitai.com/models/7?a=1&b=2.")
        assert "<script>" not in html and "&lt;script&gt;" in html
        assert html.count("<p>") == 2 and "<br>" in html
        assert '<a href="https://civitai.com/models/7?a=1&amp;b=2"' in html
        assert html.endswith(".</p>"), "sentence punctuation stays outside the link"


# ------------------------------------------------------------- MOSS layouts


class TestMossLayouts:
    def test_a_moss_layout_prompt_is_read_and_kept_verbatim(self):
        import mc_lora_civitai as civitai

        layout = {
            "summary": {"caption": "A cozy nook (dusk)", "meta": ["photo"], "canvas": "single"},
            "items": [
                {"label": "object", "caption": "armchair", "bbox": [-10, -10, 10, 10]},
                {"label": "text", "caption": "a sign", "text": "READ", "bbox": [0, 0, 5, 5]},
            ],
            "bbox_coord_space": "abs_centralized",
            "bbox_format": "yxyx",
        }
        compiled = layouts.escape_parens(json.dumps(layout, ensure_ascii=False, separators=(",", ":")))
        assert "\\(dusk\\)" in compiled

        parsed = layouts.parse(compiled)
        assert parsed.caption == "A cozy nook (dusk)"
        assert [item.label for item in parsed.items] == ["object", "text"] and parsed.items[1].text == "READ"
        assert parsed.describe() == "1 object, 1 text"
        assert layouts.label(compiled, 60).startswith("[layout] A cozy nook")
        for plain in ("a plain prompt (with parens)", "{not json}", '{"a": 1}', ""):
            assert layouts.parse(plain) is None, plain
        assert layouts.label("a plain prompt", 60) == "a plain prompt"

        # the API passing the layout on as an object gets Forge's compiled line back
        assert civitai.extract_positive_prompt({"meta": {"prompt": layout}}) == compiled
        assert civitai.extract_positive_prompt({"meta": {"prompt": {"caption": "just (this)"}}}) == "just \\(this\\)"
        # one that arrives as text is passed through untouched
        assert civitai.extract_positive_prompt({"meta": {"prompt": compiled}}) == compiled
        # and an object that is no layout is still ignored
        assert civitai.extract_positive_prompt({"meta": {"prompt": {"nodes": [1, 2]}}}) == ""


# ------------------------------------------------------------- the fragments


MOSS_ROW = {
    "safetensor_file": "moss/mossLight_v1.safetensors",
    "sha256": "ab" * 32,
    "civitai_model_name": "Moss light",
    "civitai_version_name": "v1.0",
    "base_model": "MOSS Image",
    "civitai_creator": "kohaku",
    "civitai_model_id": "11",
    "civitai_version_id": "22",
    "civitai_url": "https://civitai.com/models/11?modelVersionId=22",
    "trigger_words": "rim light, backlit",
    "status": "ok",
    "note": "fine",
    "model_description": "About it.",
    "version_description": "First release.",
    "negative_prompt": "flat",
    "positive_prompt_1": "a prompt",
}


class TestTheFragments:
    """What mc_lora_render writes for the dialog, with the classes style.css
    and the script name."""

    def test_the_header_and_the_about_column_render_the_record(self):
        record = parse_row(dict(MOSS_ROW), source="civitai-moss.csv", row_number=2)

        header = render.header_html(records=[record], how="SHA256")
        assert "Moss light" in header and "Open on Civitai" in header
        assert all(text in header for text in ("v1.0", "MOSS Image", "by kohaku", "civitai-moss.csv (row 2)"))

        facts = render.facts_html([record], identity={"SHA256": "ab" * 32, "Forge hash": "abcdef123456"}, architecture="MOSS Image")
        assert "Creator" in facts and "tensor names" in facts and "abcdef123456" in facts

        chips = render.triggers_html(["rim light", "<b>x</b>"], target="t_activation", active={"rim light"})
        assert 'data-mc-lora-target="t_activation"' in chips and chips.count("mc-lora-word-on") == 1
        assert "<b>x</b>" not in chips and "&lt;b&gt;x&lt;/b&gt;" in chips
        assert 'data-mc-lora-action="copy-word"' in render.triggers_html(["a"]), "without a target a chip copies"

        side = render.descriptions_html([(record.title, record.source, record.description, record.version_description)])
        assert "Description" in side and "Version notes" in side and "mc-lora-clamp" not in side
        long_text = "word " * 400
        clamped = render.text_section_html("Description", long_text)
        assert "mc-lora-clamp" in clamped and "Show more" in clamped
        assert 'data-mc-lora-action="append-negative"' in render.negative_html(record.negative_prompts, tab="img2img")
        assert render.fields_html([record]).startswith("<details") and "My" not in render.fields_html([record])

        unresolved = parse_row({"safetensor_file": "x.safetensors", "sha256": "cd" * 32, "status": "not_found", "note": "SHA256 not found on Civitai"}, source="c.csv", row_number=2)
        html = render.header_html(records=[unresolved], how="SHA256")
        assert "Civitai had nothing" in html and "not_found" in html and "SHA256 not found" in html
        assert "mc-lora-head-title" not in html, "no title for a LoRA Civitai did not name"
        assert "No CSV row matched" in render.header_html(records=[], how="")

    def test_a_column_title_names_the_column_and_what_is_under_it(self):
        assert render.column_title_html("Civitai") == '<div class="mc-lora-col-title">Civitai</div>'
        titled = render.column_title_html("Civitai", "3 trigger words, <6> prompts")
        assert titled.startswith('<div class="mc-lora-col-title">Civitai <span class="mc-lora-muted">')
        assert "&lt;6&gt;" in titled and "<6>" not in titled

    def test_the_image_status_line_reads_as_a_sentence(self):
        saved = render.image_status_html(downloaded=True, name="civitai-9001.jpeg", url="https://image.civitai.com/x/9001.jpeg")
        assert "Saved as <code>civitai-9001.jpeg</code>" in saved
        assert "(image.civitai.com)</span>." in saved and "</code>." not in saved
        pending = render.image_status_html(downloaded=False, name="", url="https://image.civitai.com/x/9001.jpeg")
        assert "not downloaded yet <span class=\"mc-lora-muted\">(image.civitai.com)</span>." in pending
        assert render.image_status_html(downloaded=False, name="", url="").endswith("yet.</div>")

    def test_prompt_cards_carry_the_exact_prompt_and_escape_it(self):
        layout_prompt = layouts.escape_parens(json.dumps({"summary": {"caption": "Two friends (laughing)"}, "items": []}, separators=(",", ":")))
        cards = [
            {"index": 0, "prompt": "a <b>bold</b> (prompt)", "thumb": "", "saved": False, "video": False, "url": "http://x/1.jpeg"},
            {"index": 1, "prompt": layout_prompt, "thumb": "data:image/webp;base64,AAAA", "saved": True, "video": False, "url": "http://x/2.jpeg"},
        ]
        html = render.prompt_cards_html(cards, total=5, tab="txt2img", bridge="pick_id")

        assert html.count('class="mc-lora-card"') == 2
        assert 'data-mc-lora-bridge="pick_id"' in html and 'data-mc-lora-tab="txt2img"' in html
        assert "<b>bold</b>" not in html and "&lt;b&gt;bold&lt;/b&gt;" in html
        assert "Two friends (laughing)" in html and "mc-lora-badge-inline" in html, "a layout shows its caption"
        assert 'data-mc-lora-action="download"' in html, "with images around, an unsaved card offers its download"
        assert "…and 3 more" in html

        sources = [html_module.unescape(text) for text in re.findall(r'<textarea class="mc-lora-copy-source"[^>]*>(.*?)</textarea>', html, re.S)]
        assert sources == [cards[0]["prompt"], layout_prompt], "Copy/Append/Replace act on the prompt exactly"

        # with no image saved anywhere, the cards stay text-only
        bare = render.prompt_cards_html([dict(cards[0])], total=1, tab="txt2img", bridge="pick_id")
        assert "mc-lora-card-media" not in bare

    def test_the_summary_names_the_scope_of_every_folder(self):
        html = render.preset_folder_summary_html(
            "zit",
            [("Lora", [("/models/Lora/zit", True, True), ("/models/Lora/gone", False, False)]), ("Checkpoints", [])],
        )
        assert "zit" in html
        assert "this folder and its subfolders" in html
        assert "this folder only" in html
        assert "folder is missing" in html

        assert "Nothing is assigned" in render.preset_folder_summary_html("xl", [("Lora", [])])

    def test_nothing_rendered_carries_the_standalones_names(self):
        """Every class and data attribute is mc-lora-; the script and the
        stylesheet look for nothing else."""
        record = parse_row(dict(MOSS_ROW), source="civitai-moss.csv", row_number=2)
        html = "".join([
            render.header_html(records=[record], how="SHA256"),
            render.facts_html([record], identity={"SHA256": "ab" * 32}),
            render.triggers_html(["a"], target="t", active=set()),
            render.descriptions_html([(record.title, record.source, "x " * 900, "y")]),
            render.negative_html(["flat"], tab="txt2img"),
            render.fields_html([record]),
            render.raw_html("{}"),
            render.prompt_cards_html([{"index": 0, "prompt": "p", "thumb": "", "saved": False, "video": False, "url": "u"}], total=1, tab="txt2img", bridge="b"),
            render.prompts_head_html(1, 0),
            render.library_summary_html({"files": 1, "rows": 1, "resolved": 1, "with_prompts": 1, "with_images": 0, "directory": "/x"}),
            render.job_html([]),
        ])
        assert "eld-" not in html and "eld_" not in html
        assert re.search(r'class="[^"]*\bmc-lora-', html)
