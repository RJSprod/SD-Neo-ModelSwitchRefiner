"""The Civitai fetch behind the LoRA dialog: what a scan of a folder writes,
what a refresh costs, what a MOSS-Image LoRA's own file says about it, the
jobs that run it in the background, and what the last fetch leaves behind.

Carried in with the LoRA details (docs/26). Civitai is a stub here, shaped
like the API: a version answers with a model record that has a name and a
type but no description, so asking for one costs a model-page fetch.
"""

from __future__ import annotations

import csv
import hashlib
import json
import stat
import struct
import time
from pathlib import Path

import pytest

import mc_lora_civitai as civitai
import mc_lora_hashing as hashing
import mc_lora_jobs as jobs
import mc_lora_last_used as last_used
from mc_lora_library import Library


def make_safetensors(path: Path, payload: bytes, metadata: dict | None = None) -> str:
    header = json.dumps({"__metadata__": metadata or {}}).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(struct.pack("<Q", len(header)) + header + payload)
    return hashlib.sha256(path.read_bytes()).hexdigest()


MOSS_NAMES = (
    "diffusion_model.layers.0.proj.ffn_up_gate.lora_A.weight",
    "diffusion_model.layers.0.proj.ffn_up_gate.lora_B.weight",
)


def make_lora(path: Path, names, metadata: dict | None = None) -> str:
    """A safetensors file with real tensor entries in its header."""
    header: dict = {"__metadata__": metadata or {}}
    for index, name in enumerate(names):
        header[name] = {"dtype": "F16", "shape": [1], "data_offsets": [index * 2, index * 2 + 2]}
    raw = json.dumps(header).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + b"\x00" * (2 * len(names)))
    return hashlib.sha256(path.read_bytes()).hexdigest()


class StubClient:
    """Stands in for CivitaiClient; records every hash it was asked about."""

    def __init__(self, known: set[str], **_kwargs):
        self.known = known
        self.lookups: list[str] = []
        self.model_pages: list[int] = []

    def close(self):
        pass

    def version_by_hash(self, sha256):
        self.lookups.append(sha256)
        if sha256 not in self.known:
            return None
        return {
            "id": 1,
            "modelId": 2,
            "name": "v1",
            "baseModel": "SDXL",
            "description": "<p>Version <b>notes</b>.</p>",
            "trainedWords": ["trig"],
            "files": [{"name": "f", "hashes": {"SHA256": sha256}}],
            "model": {"name": "Stub Model", "type": "LORA"},
        }

    def model_info(self, version, *, with_description=False):
        if not with_description:
            return version["model"]
        self.model_pages.append(version["modelId"])
        return {**version["model"], "description": "<p>About <b>this</b> model.</p>", "creator": {"username": "maker"}}

    def gallery_prompts(self, _version_id, *, embedded_images, max_images):
        return (civitai.PromptImage("a prompt", "https://image.example/1.jpeg", "1"),), None


@pytest.fixture
def stub(monkeypatch, lora_root):
    """Civitai knows the hashes given; the client each scan built is kept."""
    created = {}

    def install(known, client_class=StubClient):
        def factory(**kwargs):
            client = client_class(known, **kwargs)
            created["client"] = client
            return client

        monkeypatch.setattr(civitai, "CivitaiClient", factory)
        return created

    return install


def quiet(_message):
    pass


# ----------------------------------------------------------------- base models


class TestBaseModels:
    def test_base_model_filter_is_forgiving_about_spelling(self):
        assert civitai.matches_base_model("Krea2", "Krea 2")
        assert civitai.matches_base_model("Krea-2-Turbo", "Krea 2")
        assert not civitai.matches_base_model("Flux.1 D", "Krea 2")
        assert civitai.matches_base_model("anything at all", "")  # empty filter accepts everything

    def test_several_base_models_can_be_accepted(self):
        assert civitai.base_model_choices("Krea 2, MOSS Image") == ["Krea 2", "MOSS Image"]
        assert civitai.base_model_choices(" ,; ") == []
        assert civitai.matches_base_model("MOSS Image", "Krea 2, MOSS Image")
        assert civitai.matches_base_model("Krea2", "Krea 2; MOSS")
        assert not civitai.matches_base_model("Flux.1 D", "Krea 2, MOSS Image")

        assessment = civitai.assess_base_model({"baseModel": "MOSS Image"}, {}, wanted="Krea 2, MOSS Image", require_explicit=True)
        assert assessment.allowed and assessment.explicit

    def test_a_variant_type_is_not_a_conflicting_base_model(self):
        # "Standard" says which variant a version is, not which family: a LoRA
        # labelled "Other" must not be rejected because of it.
        other = civitai.assess_base_model({"baseModel": "Other", "baseModelType": "Standard"}, {}, wanted="MOSS Image", require_explicit=False)
        assert other.allowed and not other.explicit, other

        krea = civitai.assess_base_model({"baseModel": "Krea 2", "baseModelType": "Standard"}, {}, wanted="MOSS Image", require_explicit=False)
        assert not krea.allowed and "Krea 2" in krea.reason and "Standard" not in krea.reason

        assert civitai.family_label({"baseModelType": "Standard"}) == ""
        assert civitai.family_label({"baseModel": "MOSS Image", "baseModelType": "Standard"}) == "MOSS Image"

    def test_a_version_whose_hashes_disagree_is_rejected(self):
        version = {"files": [{"name": "f", "hashes": {"SHA256": "11" * 32}}]}
        allowed, reason = civitai.version_file_hash_state(version, "22" * 32)
        assert not allowed and "inconsistent" in reason

        allowed, _ = civitai.version_file_hash_state(version, "11" * 32)
        assert allowed

        # no SHA256 exposed at all: the by-hash lookup is the identity evidence
        allowed, _ = civitai.version_file_hash_state({"files": [{"name": "f"}]}, "11" * 32)
        assert allowed

    def test_the_negative_prompt_is_never_used_as_a_positive_one(self):
        assert civitai.extract_positive_prompt({"meta": {"negativePrompt": "bad"}}) == ""
        assert civitai.extract_positive_prompt({"meta": {"prompt": "good", "negativePrompt": "bad"}}) == "good"
        assert civitai.extract_positive_prompt({"meta": json.dumps({"prompt": "stringified"})}) == "stringified"


# ------------------------------------------------------------------- MOSS Image


class TestMossImage:
    def test_a_moss_lora_is_recognised_by_its_tensor_names(self, tmp_path):
        path = tmp_path / "moss.safetensors"
        make_lora(path, MOSS_NAMES, {"ss_network_module": "networks.lora"})
        metadata, names = civitai.read_safetensors_header(path)
        assert metadata == {"ss_network_module": "networks.lora"}
        assert civitai.local_architecture(names) == "MOSS Image"
        assert civitai.read_safetensors_metadata(path) == metadata

        assert civitai.local_architecture(["lora_unet_layers_3_proj_ffn_down.lora_up.weight"]) == "MOSS Image"
        for name in (
            "diffusion_model.blocks.0.attn.qkv.lora_A.weight",
            "lora_unet_double_blocks_0_img_attn_proj.lora_down.weight",
            "transformer.layers.0.attention.qkv.lora_A.weight",
            "lora_unet_layers_0_feed_forward_w1.lora_down.weight",
            "head.proj.0.weight",
        ):
            assert civitai.local_architecture([name]) == "", name

        # Civitai's label is generic: the names are explicit evidence for MOSS...
        moss = civitai.assess_base_model({"baseModel": "Other"}, {}, wanted="MOSS Image", require_explicit=True, tensor_names=MOSS_NAMES)
        assert moss.allowed and moss.explicit and "tensor names" in moss.reason
        # ...and never let a MOSS LoRA through a filter for something else.
        krea = civitai.assess_base_model({"baseModel": "Other"}, {}, wanted="Krea 2", require_explicit=True, tensor_names=MOSS_NAMES)
        assert not krea.allowed

    def test_a_fetch_uses_the_tensor_names_of_the_file(self, tmp_path, stub):
        loras = tmp_path / "loras"
        sha = make_lora(loras / "moss.safetensors", MOSS_NAMES)

        class OtherLabel(StubClient):
            def version_by_hash(self, sha256):
                version = super().version_by_hash(sha256)
                if version is not None:
                    version["baseModel"] = "Other"
                return version

        stub({sha}, OtherLabel)
        report = civitai.scan_folder(
            loras,
            tmp_path / "out.csv",
            base_model_filter="MOSS Image",
            require_explicit_label=True,
            log=quiet,
        )
        assert report.resolved == 1, report.counts
        row = next(iter(csv.DictReader((tmp_path / "out.csv").open(encoding="utf-8"))))
        assert row["status"] == "ok" and "tensor names" in row["note"]

    def test_moss_is_a_known_preset_without_forge(self):
        import mc_extra_networks_folders

        assert "moss" in mc_extra_networks_folders.FALLBACK_PRESETS
        assert "qwen21" in mc_extra_networks_folders.FALLBACK_PRESETS


# -------------------------------------------------------------------- scanning


class TestScanning:
    def test_scan_writes_a_hash_column_and_refetches_only_new_files(self, tmp_path, stub):
        loras = tmp_path / "loras"
        sha_a = make_safetensors(loras / "alpha.safetensors", b"A")
        sha_b = make_safetensors(loras / "sub" / "beta.safetensors", b"B")
        output = tmp_path / "out.csv"
        created = stub({sha_a})

        report = civitai.scan_folder(loras, output, log=quiet)
        assert report.total == 2 and report.resolved == 1
        assert len(created["client"].lookups) == 2

        written = list(csv.DictReader(output.open(encoding="utf-8")))
        assert {row["safetensor_file"] for row in written} == {"alpha.safetensors", "sub/beta.safetensors"}
        alpha = next(row for row in written if row["safetensor_file"] == "alpha.safetensors")
        assert alpha["sha256"] == sha_a, "the SHA256 column is what makes hash lookup work"
        assert alpha["civitai_model_name"] == "Stub Model"
        assert alpha["positive_prompt_1"] == "a prompt"
        assert alpha["prompt_image_url_1"] == "https://image.example/1.jpeg"
        assert alpha["prompt_image_id_1"] == "1"

        # the library indexes what was just written, by hash
        library = Library(tmp_path)
        library.reload()
        records, how = library.lookup(sha256=sha_a)
        assert how == "SHA256" and records[0].model_name == "Stub Model"
        assert records[0].entries[0].image_url == "https://image.example/1.jpeg"

        # a refresh of the same folder costs nothing and keeps every row
        report = civitai.scan_folder(loras, output, skip_hashes=library.known_sha256(), log=quiet)
        assert created["client"].lookups == [], "a refresh must not re-query anything already recorded"
        assert report.skipped == 2 and len(report.rows) == 2

        # opting in to retries picks the unresolved one back up, and only it
        retry = library.known_sha256(resolved_only=True)
        civitai.scan_folder(loras, output, skip_hashes=retry, log=quiet)
        assert created["client"].lookups == [sha_b], created["client"].lookups

        # only a genuinely new file is fetched
        sha_c = make_safetensors(loras / "gamma.safetensors", b"C")
        library.reload()
        report = civitai.scan_folder(loras, output, skip_hashes=library.known_sha256(), log=quiet)
        assert created["client"].lookups == [sha_c], "only the genuinely new LoRA is looked up"
        assert len(report.rows) == 3

    def test_rewriting_from_scratch_never_drops_files(self, tmp_path, stub):
        loras = tmp_path / "loras"
        sha_a = make_safetensors(loras / "alpha.safetensors", b"A")
        output = tmp_path / "out.csv"
        stub({sha_a})

        civitai.scan_folder(loras, output, log=quiet)
        # "skip what we have" + "rewrite from scratch" would otherwise lose rows
        report = civitai.scan_folder(loras, output, skip_hashes={sha_a}, keep_existing_rows=False, log=quiet)
        assert report.skipped == 0 and len(report.rows) == 1

    def test_the_csv_is_replaced_only_after_a_complete_write(self, tmp_path):
        output = tmp_path / "out.csv"
        civitai.write_csv(output, [{"safetensor_file": "a", "sha256": "b"}])
        before = output.read_text(encoding="utf-8")

        class Unwritable:
            def __str__(self):
                raise RuntimeError("boom halfway through the write")

        with pytest.raises(RuntimeError):
            civitai.write_csv(output, [{"safetensor_file": "a"}, {"safetensor_file": Unwritable()}])
        assert output.read_text(encoding="utf-8") == before
        assert not list(tmp_path.glob("*.tmp")), "the temp file must be cleaned up"

    def test_a_fetch_stores_the_description_and_backfills_rows_that_lack_one(self, tmp_path, stub):
        loras = tmp_path / "loras"
        sha_a = make_safetensors(loras / "alpha.safetensors", b"A")
        sha_b = make_safetensors(loras / "beta.safetensors", b"B")   # Civitai knows nothing about it
        output = tmp_path / "out.csv"
        created = stub({sha_a})

        civitai.scan_folder(loras, output, log=quiet)
        assert created["client"].model_pages == [2], "the description costs one model-page fetch"

        row = next(r for r in csv.DictReader(output.open(encoding="utf-8")) if r["sha256"] == sha_a)
        assert row["model_description"] == "About this model."
        assert row["version_description"] == "Version notes."
        assert row["civitai_creator"] == "maker", "the model page names the creator"

        library = Library(tmp_path)
        library.reload()
        records, _how = library.lookup(sha256=sha_a)
        assert records[0].description == "About this model."

        # a refresh has nothing to fill in, so it costs no requests at all
        civitai.scan_folder(loras, output, skip_hashes=library.known_sha256(require_description=True), log=quiet)
        assert created["client"].lookups == [], "nothing is missing, so nothing is re-queried"

        # a row written before descriptions existed is looked up again...
        stripped = list(csv.DictReader(output.open(encoding="utf-8")))
        for entry in stripped:
            entry["model_description"] = entry["version_description"] = ""
        civitai.write_csv(output, stripped)
        library.reload()
        assert library.known_sha256() == {sha_a, sha_b}, "plain refresh still skips everything recorded"

        civitai.scan_folder(loras, output, skip_hashes=library.known_sha256(require_description=True), log=quiet)
        assert created["client"].lookups == [sha_a], "only the resolved row missing its text is re-fetched"

        # ...and the description comes back
        row = next(r for r in csv.DictReader(output.open(encoding="utf-8")) if r["sha256"] == sha_a)
        assert row["model_description"] == "About this model."

    def test_turning_descriptions_off_costs_no_extra_request(self, tmp_path, stub):
        loras = tmp_path / "loras"
        sha_a = make_safetensors(loras / "alpha.safetensors", b"A")
        output = tmp_path / "out.csv"
        created = stub({sha_a})

        civitai.scan_folder(loras, output, fetch_descriptions=False, log=quiet)
        assert created["client"].model_pages == []
        row = next(iter(csv.DictReader(output.open(encoding="utf-8"))))
        assert row["model_description"] == "" and row["civitai_model_name"] == "Stub Model"
        assert row["civitai_creator"] == "", "only the model page names the creator"

    def test_a_refetch_keeps_hand_written_columns_but_not_stale_prompts(self):
        fresh = {"safetensor_file": "a.safetensors", "sha256": "ab" * 32, "positive_prompt_1": "new"}
        existing = {
            "safetensor_file": "a.safetensors",
            "sha256": "ab" * 32,
            "civitai_model_name": "Old Name",
            "positive_prompt_1": "old",
            "positive_prompt_2": "dropped with the version it came from",
            "My Rating": "5 stars",
        }
        merged = civitai.merge_row(existing, fresh)
        assert merged["My Rating"] == "5 stars", "a column this tool does not own survives"
        assert merged["positive_prompt_1"] == "new"
        assert "positive_prompt_2" not in merged
        assert merged.get("civitai_model_name", "") == ""
        assert civitai.merge_row(None, fresh) == fresh

    def test_only_a_key_civitai_answered_counts_as_used(self, tmp_path, stub):
        loras = tmp_path / "loras"
        sha_a = make_safetensors(loras / "alpha.safetensors", b"A")
        stub({sha_a})

        report = civitai.scan_folder(loras, tmp_path / "out.csv", log=quiet)
        assert report.api_answered, "an answered lookup is what proves the key was accepted"

        report = civitai.scan_folder(loras, tmp_path / "out.csv", skip_hashes={sha_a}, log=quiet)
        assert not report.api_answered, "a run that asked nothing proves nothing"

        class Rejecting(StubClient):
            def version_by_hash(self, sha256):
                raise civitai.CivitaiError("Civitai returned HTTP 401; check the API key")

        stub(set(), Rejecting)
        report = civitai.scan_folder(loras, tmp_path / "rejected.csv", log=quiet)
        assert report.processed == 1 and not report.api_answered

    def test_a_scan_logs_counts_never_a_path(self, tmp_path, stub, caplog):
        """CLAUDE.md's standing convention: no path and no user's file name in
        a log line. A scan says how many; the CSV says which."""
        import logging

        loras = tmp_path / "loras"
        sha_a = make_safetensors(loras / "Alpha Private Name.safetensors", b"A")
        stub({sha_a})
        with caplog.at_level(logging.INFO, logger="model_chain"):
            job = jobs.submit_scan(loras, api_key="", recursive=False, incremental=False, retry_unresolved=False,
                                   backfill_text=True, fetch_descriptions=True, base_model_filter="",
                                   require_explicit_label=False, request_delay=0.0, timeout=5.0, max_images=0,
                                   keep_existing_rows=True)
            deadline = time.time() + 15
            while jobs.MANAGER.get(job.id).snapshot().active and time.time() < deadline:
                time.sleep(0.05)
        assert jobs.MANAGER.get(job.id).snapshot().state == jobs.DONE
        lines = [record.getMessage() for record in caplog.records if record.name == "model_chain"]
        assert lines, "the job is announced"
        assert not any("Alpha Private Name" in line or str(tmp_path) in line for line in lines), lines


# ------------------------------------------------------------------------ jobs


class TestJobs:
    def test_jobs_report_progress_failure_and_cancellation(self):
        manager = jobs.JobManager()

        def counting(job):
            for index in range(4):
                if job.cancelled:
                    break
                job.set_progress(index + 1, 4, f"step {index + 1}")
                job.log_line(f"step {index + 1}")
                time.sleep(0.02)
            return "counted"

        manager.submit("test", "counting", counting)
        deadline = time.time() + 5
        while manager.snapshots()[0].active and time.time() < deadline:
            time.sleep(0.02)

        snapshot = manager.snapshots()[0]
        assert snapshot.state == jobs.DONE and snapshot.percent == 100
        assert snapshot.summary == "counted" and len(snapshot.log) == 4

        def failing(_job):
            raise ValueError("nope")

        manager.submit("test", "failing", failing)
        deadline = time.time() + 5
        while manager.snapshots()[0].active and time.time() < deadline:
            time.sleep(0.02)
        assert manager.snapshots()[0].state == jobs.FAILED
        assert "ValueError: nope" in manager.snapshots()[0].error

        def looping(job):
            while not job.cancelled:
                time.sleep(0.02)
            return "stopped"

        manager.submit("test", "looping", looping)
        deadline = time.time() + 5
        while not manager.has_active() and time.time() < deadline:
            time.sleep(0.02)
        assert manager.cancel_all() == 1
        deadline = time.time() + 5
        while manager.has_active() and time.time() < deadline:
            time.sleep(0.02)
        assert manager.snapshots()[0].state == jobs.CANCELLED

        assert manager.clear_finished() == 3 and manager.snapshots() == []

    def test_jobs_run_one_at_a_time(self):
        manager = jobs.JobManager()
        concurrent = []
        running = []

        def watcher(_job):
            running.append(1)
            concurrent.append(len(running))
            time.sleep(0.05)
            running.pop()
            return ""

        for _ in range(4):
            manager.submit("test", "watcher", watcher)

        deadline = time.time() + 10
        while manager.has_active() and time.time() < deadline:
            time.sleep(0.02)
        assert concurrent == [1, 1, 1, 1], concurrent

    def test_the_jobs_view_renders_every_state(self):
        import mc_lora_render as render

        manager = jobs.JobManager()
        manager.submit("scan", "a scan", lambda job: "3 LoRAs")
        deadline = time.time() + 5
        while manager.has_active() and time.time() < deadline:
            time.sleep(0.02)
        html = render.job_html(manager.snapshots())
        assert "a scan" in html and "3 LoRAs" in html and "mc-lora-job" in html
        assert "No scans have been started yet" in render.job_html([])


# ------------------------------------------------------- remembered values


class TestWhatTheLastFetchLeaves:
    def test_the_key_and_folder_of_the_last_fetch_are_remembered(self, lora_root, host):
        folder = lora_root.parent / "loras"
        folder.mkdir()

        last_used.remember_api_key("  secret-key  ")
        last_used.remember_scan_folder(folder)

        assert last_used.remembered_api_key() == "secret-key"
        assert last_used.remembered_scan_folder() == str(folder)

        path = lora_root / "last_used.json"
        assert path.is_file(), "under the data root, beside the folder assignments"
        # a fresh reader sees both - which is all the next session does
        reread = last_used.LastUsedStore(path)
        assert reread.get(last_used.API_KEY) == "secret-key"
        assert reread.get(last_used.SCAN_FOLDER) == str(folder)

        # a run without a key never clears the one that worked
        last_used.remember_api_key("")
        assert last_used.get_store().get(last_used.API_KEY) == "secret-key"

        import os

        if os.name == "posix":
            assert stat.S_IMODE(path.stat().st_mode) & 0o077 == 0, "the file holds an API key"

    def test_a_folder_that_has_gone_away_is_not_offered_but_is_kept(self, lora_root, host):
        folder = lora_root.parent / "loras"
        folder.mkdir()

        last_used.remember_scan_folder(folder)
        folder.rmdir()

        # an unplugged drive must not become a default that cannot be scanned
        assert last_used.remembered_scan_folder() == ""
        # but the choice is kept, and is the default again when it comes back
        assert last_used.get_store().get(last_used.SCAN_FOLDER) == str(folder)
        folder.mkdir()
        assert last_used.remembered_scan_folder() == str(folder)

    def test_turning_remembering_off_hides_the_values_and_drops_the_key(self, lora_root, host, monkeypatch):
        folder = lora_root.parent / "loras"
        folder.mkdir()

        last_used.remember_api_key("secret-key")
        last_used.remember_scan_folder(folder)

        monkeypatch.setattr(host.shared.opts, "model_chain_lora_remember_api_key", False, raising=False)
        monkeypatch.setattr(host.shared.opts, "model_chain_lora_remember_scan_folder", False, raising=False)
        assert last_used.remembered_api_key() == ""
        assert last_used.remembered_scan_folder() == ""
        last_used.remember_api_key("another-key")
        assert last_used.get_store().get(last_used.API_KEY) == "secret-key", "nothing is stored while it is off"
        last_used.forget_api_key()  # what turning the setting off does

        monkeypatch.setattr(host.shared.opts, "model_chain_lora_remember_api_key", True, raising=False)
        monkeypatch.setattr(host.shared.opts, "model_chain_lora_remember_scan_folder", True, raising=False)
        assert last_used.remembered_api_key() == ""
        assert last_used.remembered_scan_folder() == str(folder), "only the key is forgotten"

    def test_the_setting_forgets_the_key_the_moment_it_is_turned_off(self, lora_root, host, monkeypatch):
        import mc_lora_settings

        last_used.remember_api_key("secret-key")
        monkeypatch.setattr(host.shared.opts, "model_chain_lora_remember_api_key", False, raising=False)
        mc_lora_settings._forget_key_when_off()
        monkeypatch.setattr(host.shared.opts, "model_chain_lora_remember_api_key", True, raising=False)
        assert last_used.remembered_api_key() == ""


def test_hashing_logs_never_name_the_file(tmp_path, lora_root, caplog):
    """A LoRA that cannot be hashed is reported by its failure, not its name."""
    import logging
    import os

    path = tmp_path / "Secret Project.safetensors"
    path.write_bytes(b"x")
    os.chmod(path, 0)
    try:
        with caplog.at_level(logging.INFO, logger="model_chain"):
            result = hashing.sha256_file(path)
    finally:
        os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)
    if os.geteuid() == 0:
        pytest.skip("root reads an unreadable file")
    assert result is None
    assert not any("Secret Project" in record.getMessage() for record in caplog.records)
