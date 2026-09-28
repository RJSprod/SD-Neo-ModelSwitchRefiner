"""Saved prompt caches: a conversation's llama.cpp slot, kept on disk across restarts.

From one user's logs, five days on an Intel Arc: eighteen WebUI restarts, and
every first reply after one read the whole conversation -- four to six thousand
tokens at 50 to 90 a second -- before its first word. llama-server can write a
slot to a file and read it back, and its build b10621 does that for a slot with
pictures in it too. These tests hold the pieces that make that safe to use:

* the start carries ``--slot-save-path`` only when saving is on, the model can
  be resumed exactly, the build has the flag and the folder exists -- a missing
  folder is a start that fails, and a sliding-window model's saved slot holds
  one window of positions (checked against a real llama-server; see
  ``TestOnlyWhatLlamaCppCanResumeExactly``);
* such a model's file is read back only when the new prompt extends it token
  for token (``TestAWindowedModelIsReadBackOnlyWhenExact``);
* a file is filed under the conversation *and* the server that wrote it, so a
  different model, projector, build or cache is never offered another's file;
* a file is read into a slot nobody has used, once per conversation per
  process, and the reply is sent to that slot;
* the slot that answered is the one saved, as llama-server itself reports it;
* the request body is the vendored client's with two fields added, and the
  vendored client, every other caller and every test double are untouched;
* the folder stays within its budget, the newest file first.

The fake llama-server answers the slot routes and the chat stream in the shapes
b10621's ``server-task.cpp`` writes them. Every test here was checked against
the decision it guards by reverting that decision.
"""

from __future__ import annotations

import json
import os
import struct
import threading
import time
import types
from pathlib import Path

import httpx
import pytest

import mc_gguf
import mc_llm_context as ctx
import mc_llm_paths
import mc_llm_runtime as runtime
import mc_llm_sessions as sessions
import mc_llm_slot_cache as cache
from test_gguf import GEMMA4_KEYS, _text, _u32, gemma4_header, write_gguf

GB = 1024 ** 3
MB = 1024 ** 2


# --------------------------------------------------------------------------- #
# A llama-server that answers the slot routes and the chat stream
# --------------------------------------------------------------------------- #


class FakeResponse:
    def __init__(self, status=200, body=None, lines=()):
        self.status_code = status
        self._body = body
        self._lines = list(lines)

    def json(self):
        if self._body is None:
            raise ValueError("no body")
        return self._body

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError(f"HTTP {self.status_code}", request=None, response=None)

    def iter_lines(self):
        yield from self._lines

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class FakeLlamaServer:
    """The routes this feature touches, answered as llama.cpp b10621 answers them."""

    def __init__(self, slots=6):
        self.slots = [{"id": index, "n_ctx": 8192, "speculative": False,
                       "is_processing": False} for index in range(slots)]
        self.requests: list[tuple] = []
        self.restore_status = 200
        self.save_status = 200
        self.answering = 3
        """The slot llama-server picks for a request that does not name one."""
        self.pieces = ("Hello", " there")
        self.saved: list[tuple] = []

    def handle(self, method, url, body, headers):
        self.requests.append((method, url, body, dict(headers or {})))
        path = url[url.index("/", len("http://")):]
        if method == "GET" and path == "/slots":
            return FakeResponse(200, [dict(slot) for slot in self.slots])
        if method == "POST" and path.startswith("/slots/"):
            slot = int(path[len("/slots/"):].split("?")[0])
            if path.endswith("action=restore"):
                if self.restore_status != 200:
                    return FakeResponse(self.restore_status, {"error": {
                        "code": self.restore_status, "type": "invalid_request_error",
                        "message": "Unable to restore slot: No available space in KV cache "
                                   "or invalid slot save file"}})
                return FakeResponse(200, {"id_slot": slot, "filename": body["filename"],
                                          "n_restored": 4861, "n_read": 212 * MB,
                                          "timings": {"restore_ms": 812.5}})
            if path.endswith("action=save"):
                if self.save_status != 200:
                    return FakeResponse(self.save_status, {"error": {
                        "code": self.save_status, "message": "Unable to save slot"}})
                self.saved.append((slot, body["filename"]))
                return FakeResponse(200, {"id_slot": slot, "filename": body["filename"],
                                          "n_saved": 4861, "n_written": 212 * MB,
                                          "timings": {"save_ms": 301.0}})
        if method == "POST" and path == "/v1/chat/completions":
            return FakeResponse(200, None, self.chat(body))
        if method == "POST" and path == "/apply-template":
            return FakeResponse(200, {"prompt": rendered(body["messages"])})
        if method == "POST" and path == "/tokenize":
            return FakeResponse(200, {"tokens": tokenized(body["content"],
                                                          body.get("add_special", False))})
        return FakeResponse(404, {"error": {"message": "not found"}})

    def chat(self, body):
        slot = body.get("id_slot", -1)
        slot = self.answering if slot == -1 else slot
        lines = []
        for piece in self.pieces:
            lines.append("data: " + json.dumps({
                "choices": [{"index": 0, "delta": {"content": piece}, "finish_reason": None}],
                "object": "chat.completion.chunk"}))
            lines.append("")
        final = {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                 "object": "chat.completion.chunk"}
        if body.get("verbose"):
            # server-task.cpp: the first delta of the final result carries
            # __verbose, cut down to response_fields when it was given.
            verbose = {"id_slot": slot, "content": "".join(self.pieces), "prompt": "..."}
            fields = body.get("response_fields")
            final["__verbose"] = ({key: verbose[key] for key in fields if key in verbose}
                                  if fields else verbose)
        lines.append("data: " + json.dumps(final))
        lines.append("")
        lines.append("data: [DONE]")
        return lines

    def posted(self, action):
        return [(url, body) for method, url, body, _ in self.requests
                if method == "POST" and url.endswith(f"action={action}")]

    def chats(self):
        return [body for method, url, body, _ in self.requests
                if url.endswith("/v1/chat/completions")]


BOS = 2


def rendered(messages) -> str:
    """The fake server's chat template: every message's text, in order."""
    return "".join(f"<{message['role']}>{message['content']}" for message in messages)


def tokenized(text: str, add_special: bool) -> list[int]:
    """The fake server's tokenizer: one token per character, BOS in front."""
    return ([BOS] if add_special else []) + [ord(character) for character in text]


def slot_file(path: Path, tokens, pictures: int = 0, magic=None, version=None,
              packed_version=None, plain=False) -> Path:
    """A slot file in llama.cpp b10621's layout: header, packed tokens, state."""
    if plain:
        body = list(tokens)
    else:
        body = [cache.NULL_TOKEN, cache.PACKED_VERSION if packed_version is None
                else packed_version, len(tokens), *tokens, pictures]
        body += [7] * pictures + [4, 1] * pictures
    head = struct.pack("<III", cache.SEQUENCE_MAGIC if magic is None else magic,
                       cache.SEQUENCE_VERSION if version is None else version, len(body))
    path.write_bytes(head + struct.pack(f"<{len(body)}i", *body) + b"\0" * 64)
    return path


@pytest.fixture
def server(monkeypatch):
    fake = FakeLlamaServer()

    class FakeHttp:
        def __init__(self, *args, headers=None, timeout=None, **kwargs):
            self.headers = dict(headers or {})

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def get(self, url, **kwargs):
            return fake.handle("GET", url, None, self.headers)

        def post(self, url, json=None, **kwargs):
            return fake.handle("POST", url, json, self.headers)

        def stream(self, method, url, json=None, **kwargs):
            return fake.handle(method, url, json, self.headers)

    monkeypatch.setattr(httpx, "Client", FakeHttp)
    return fake


@pytest.fixture
def folder(tmp_path, monkeypatch, host):
    monkeypatch.setattr(mc_llm_paths, "data_root", lambda: tmp_path)
    made = tmp_path / cache.DIRNAME
    made.mkdir()
    return made


def a_model(where: Path, name="model.gguf", window=0) -> Path:
    """A GGUF header: a small attention model, with a sliding window if asked."""
    entries = [_text("general.architecture", "llama"), _u32("llama.block_count", 4),
               _u32("llama.context_length", 8192), _u32("llama.embedding_length", 256),
               _u32("llama.attention.head_count", 4), _u32("llama.attention.head_count_kv", 2)]
    if window:
        entries.append(_u32("llama.attention.sliding_window", window))
    return write_gguf(where / name, b"".join(entries), len(entries), padding=1024)


def a_server(folder, identity="abc123abc123", key="secret-key", port=58915):
    return cache.Server(f"http://127.0.0.1:{port}", key, folder, identity)


def saved_file(folder, running, conversation="Ada/thread-1", size=1024, age=0.0):
    path = folder / running.file_name(conversation)
    path.write_bytes(b"\0" * size)
    if age:
        stamp = time.time() - age
        os.utime(path, (stamp, stamp))
    return path


def wait_for(condition, seconds=5.0):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.01)
    return condition()


# --------------------------------------------------------------------------- #
# The start
# --------------------------------------------------------------------------- #


class TestTheStartCarriesTheFolder:
    """``--slot-save-path`` is what makes a server able to save at all, and a
    path that is not a directory is a start llama-server refuses."""

    LISTS_IT = "  --slot-save-path PATH   path to save slot kv cache (default: disabled)\n"

    @pytest.fixture(autouse=True)
    def forget(self, host):
        runtime._capabilities.clear()
        yield
        runtime._capabilities.clear()

    @pytest.fixture
    def build(self, tmp_path, monkeypatch):
        executable = tmp_path / "llama-server"
        executable.write_text("")

        def announce(text, model=None):
            monkeypatch.setattr(
                runtime.subprocess, "run",
                lambda *args, **kwargs: types.SimpleNamespace(stdout=text, stderr=""))
            return runtime.Config(
                runtime=executable, model=model or a_model(tmp_path), mmproj=None,
                gpu_index=0, device="SYCL0", gpu_layers="all", context_size=8192,
                context_mode="fixed", context_buffer_gb=4.0, kv_type_k="f16",
                kv_type_v="f16")

        return announce

    def test_a_model_with_a_sliding_window_is_saved_only_on_the_full_cache(
            self, build, tmp_path, monkeypatch):
        monkeypatch.setattr(mc_llm_paths, "data_root", lambda: tmp_path / "root")
        configuration = build(self.LISTS_IT, model=a_model(tmp_path, "windowed.gguf",
                                                            window=1024))

        assert cache.launch_flags(configuration) == []
        assert not (tmp_path / "root" / cache.DIRNAME).exists()
        assert cache.launch_flags(configuration, full_cache=True)[0] == cache.SLOT_SAVE_FLAG

    def test_a_model_whose_header_cannot_be_read_is_not_saved(self, build, tmp_path,
                                                              monkeypatch):
        monkeypatch.setattr(mc_llm_paths, "data_root", lambda: tmp_path / "root")
        broken = tmp_path / "broken.gguf"
        broken.write_bytes(b"not a gguf")

        assert cache.launch_flags(build(self.LISTS_IT, model=broken)) == []

    def test_a_build_that_lists_it_is_given_the_folder_and_the_folder_is_made(
            self, build, tmp_path, monkeypatch):
        monkeypatch.setattr(mc_llm_paths, "data_root", lambda: tmp_path / "root")
        configuration = build(self.LISTS_IT)

        flags = cache.launch_flags(configuration)

        assert flags == [cache.SLOT_SAVE_FLAG, str(tmp_path / "root" / cache.DIRNAME)]
        assert (tmp_path / "root" / cache.DIRNAME).is_dir()

    def test_a_build_that_does_not_list_it_starts_as_it_always_did(
            self, build, tmp_path, monkeypatch):
        monkeypatch.setattr(mc_llm_paths, "data_root", lambda: tmp_path / "root")
        configuration = build("  --swa-full\n")

        assert cache.launch_flags(configuration) == []
        assert not (tmp_path / "root" / cache.DIRNAME).exists()

    def test_the_setting_off_or_no_disk_to_spend_saves_nothing(self, build, tmp_path,
                                                               monkeypatch, host):
        monkeypatch.setattr(mc_llm_paths, "data_root", lambda: tmp_path / "root")
        configuration = build(self.LISTS_IT)

        host.shared.opts.set(cache.OPT_SAVE, False)
        assert cache.launch_flags(configuration) == []
        host.shared.opts.set(cache.OPT_SAVE, True)
        host.shared.opts.set(cache.OPT_BUDGET_GB, 0)
        assert cache.launch_flags(configuration) == []

    def test_a_folder_that_cannot_be_made_leaves_the_flag_off(self, build, tmp_path,
                                                              monkeypatch):
        blocked = tmp_path / "a-file"
        blocked.write_text("not a directory")
        monkeypatch.setattr(mc_llm_paths, "data_root", lambda: blocked)
        configuration = build(self.LISTS_IT)

        assert cache.launch_flags(configuration) == []

    def test_the_launch_line_carries_it_beside_the_placement_flags(self, build, tmp_path,
                                                                   monkeypatch):
        monkeypatch.setattr(mc_llm_paths, "data_root", lambda: tmp_path / "root")
        configuration = build(self.LISTS_IT + "  -cms, --checkpoint-min-step N\n")

        flags = runtime._launch_flags(configuration, ctx.Placement(uma=True), None)

        assert flags[-2:] == [cache.SLOT_SAVE_FLAG, str(tmp_path / "root" / cache.DIRNAME)]
        assert runtime.CHECKPOINT_SPACING_FLAG in flags

    def test_a_refusal_of_it_is_not_blamed_on_the_model(self):
        assert cache.SLOT_SAVE_FLAG in runtime.OPTIONAL_FLAGS
        said = runtime.read_failure(
            'error while handling argument "--slot-save-path": not a directory: C:\\x\n')
        assert said.bad_argument == cache.SLOT_SAVE_FLAG


class TestOnlyWhatLlamaCppCanResumeExactly:
    """Checked against llama-server b10621 with a real start, save, restart and
    restore: a slot file keeps only the window's positions of a sliding-window
    model, so a restore was read in full on the window cache every time, and on
    the full cache it resumed exactly only where the new prompt extended the
    saved one. A model without a window resumed exactly where a running server
    would have. So a model with a window is saved only on the full cache (and,
    for Gemma 4's template, with its thought marker kept -- see
    ``tests/test_llm_template.py``), and never on the window cache."""

    def test_a_model_without_a_window_is_resumable(self, tmp_path):
        assert cache.resumable(a_model(tmp_path)) is True

    def test_the_window_key_says_it_has_one(self, tmp_path):
        model = a_model(tmp_path, window=4096)

        assert mc_gguf.describe(model).sliding_window == 4096
        assert cache.resumable(model) is False

    def test_the_per_block_pattern_alone_says_it_too(self, tmp_path):
        model = write_gguf(tmp_path / "gemma4.gguf", gemma4_header(), GEMMA4_KEYS)

        assert mc_gguf.describe(model).sliding_window == 0
        assert cache.resumable(model) is False

    def test_a_file_that_is_not_there_or_not_a_model_is_not(self, tmp_path):
        broken = tmp_path / "broken.gguf"
        broken.write_bytes(b"GGUF but not really")

        assert cache.resumable(tmp_path / "missing.gguf") is False
        assert cache.resumable(broken) is False
        assert cache.resumable(None) is False


class TestWhatAFileIsValidFor:
    """A file is filed under the server that wrote it. A different model, build,
    projector, kind of window or cache type is a different identity, and a
    restore across them would be refused at best and subtly wrong at worst."""

    @pytest.fixture
    def files(self, tmp_path):
        made = {}
        for name in ("llama-server", "model.gguf", "other.gguf", "mmproj.gguf"):
            made[name] = tmp_path / name
            made[name].write_bytes(b"x" * 10)
        return made

    def test_the_same_server_is_the_same_identity(self, files):
        first = cache.identity(files["llama-server"], files["model.gguf"], None, False, "")
        again = cache.identity(files["llama-server"], files["model.gguf"], None, False, "")

        assert first == again
        assert len(first) == 12

    def test_each_part_is_a_different_identity(self, files):
        base = cache.identity(files["llama-server"], files["model.gguf"], None, False, "")

        assert cache.identity(files["llama-server"], files["other.gguf"], None, False,
                              "") != base
        assert cache.identity(files["llama-server"], files["model.gguf"],
                              files["mmproj.gguf"], False, "") != base
        assert cache.identity(files["llama-server"], files["model.gguf"], None, True,
                              "") != base
        assert cache.identity(files["llama-server"], files["model.gguf"], None, False,
                              "q8_0/q8_0") != base

    def test_a_model_replaced_in_place_is_a_different_identity(self, files):
        base = cache.identity(files["llama-server"], files["model.gguf"], None, False, "")
        files["model.gguf"].write_bytes(b"y" * 11)

        assert cache.identity(files["llama-server"], files["model.gguf"], None, False,
                              "") != base

    def test_a_new_build_is_a_different_identity(self, files):
        base = cache.identity(files["llama-server"], files["model.gguf"], None, False, "")
        stamp = time.time() + 60
        os.utime(files["llama-server"], (stamp, stamp))

        assert cache.identity(files["llama-server"], files["model.gguf"], None, False,
                              "") != base

    def test_the_runtime_reads_the_window_off_the_line_it_started(self, files, tmp_path):
        configuration = runtime.Config(
            runtime=files["llama-server"], model=files["model.gguf"], mmproj=None,
            gpu_index=0, device="SYCL0", gpu_layers="all", context_size=8192,
            context_mode="fixed", context_buffer_gb=4.0, kv_type_k="f16", kv_type_v="f16")
        process = types.SimpleNamespace(port=58915, api_key="k")
        placement = ctx.Placement(uma=True)
        folder = [cache.SLOT_SAVE_FLAG, str(tmp_path)]

        window = runtime._saved_caches_for(process, folder, files["llama-server"],
                                           configuration, placement, None)
        full = runtime._saved_caches_for(process, [runtime.FULL_ATTENTION_WINDOW_FLAG] + folder,
                                         files["llama-server"], configuration, placement,
                                         None)

        assert window is not None and full is not None
        assert window.identity != full.identity
        assert window.identity == cache.identity(files["llama-server"], files["model.gguf"],
                                                 None, False, "")

    def test_a_start_without_the_folder_cannot_save(self, files):
        process = types.SimpleNamespace(port=58915, api_key="k")

        assert cache.started(process, ["--swa-full"], "abc") is None

    def test_a_start_with_the_folder_saves_there_on_its_own_port_and_key(self, tmp_path):
        process = types.SimpleNamespace(port=58915, api_key="k")

        found = cache.started(process, [cache.SLOT_SAVE_FLAG, str(tmp_path)], "abc")

        assert found.base_url == "http://127.0.0.1:58915"
        assert found.api_key == "k"
        assert found.folder == tmp_path
        assert found.identity == "abc"


# --------------------------------------------------------------------------- #
# Restoring
# --------------------------------------------------------------------------- #


class TestRestoring:
    """A file is read into a slot nobody has used since the server started, at
    most once per conversation per process, and never over another's cache."""

    def test_no_file_asks_llama_server_nothing(self, server, folder):
        running = a_server(folder)

        assert running.restore("Ada/thread-1") is None
        assert server.requests == []

    def test_a_file_is_read_into_the_first_unused_slot(self, server, folder):
        running = a_server(folder)
        path = saved_file(folder, running, age=3600)
        before = path.stat().st_mtime

        slot = running.restore("Ada/thread-1")

        assert slot == 0
        assert server.posted("restore") == [
            ("http://127.0.0.1:58915/slots/0?action=restore", {"filename": path.name})]
        assert path.stat().st_mtime > before

    def test_the_request_carries_the_servers_own_key(self, server, folder):
        running = a_server(folder, key="per-start-key")
        saved_file(folder, running)

        running.restore("Ada/thread-1")

        assert all(headers.get("Authorization") == "Bearer per-start-key"
                   for _, _, _, headers in server.requests)

    def test_a_slot_that_has_run_a_task_or_is_busy_is_left_alone(self, server, folder):
        server.slots[0]["id_task"] = 12
        server.slots[0]["n_prompt_tokens"] = 900
        server.slots[1]["is_processing"] = True
        running = a_server(folder)
        saved_file(folder, running)

        assert running.restore("Ada/thread-1") == 2

    def test_no_unused_slot_means_no_restore(self, server, folder):
        for slot in server.slots:
            slot["id_task"] = 7
        running = a_server(folder)
        saved_file(folder, running)

        assert running.restore("Ada/thread-1") is None
        assert server.posted("restore") == []

    def test_once_per_conversation_per_process(self, server, folder):
        running = a_server(folder)
        saved_file(folder, running)

        assert running.restore("Ada/thread-1") == 0
        assert running.restore("Ada/thread-1") is None
        assert len(server.posted("restore")) == 1

    def test_two_conversations_are_never_read_into_one_slot(self, server, folder):
        running = a_server(folder)
        saved_file(folder, running, "Ada/thread-1")
        saved_file(folder, running, "Ada/thread-2")

        assert running.restore("Ada/thread-1") == 0
        assert running.restore("Ada/thread-2") == 1

    def test_a_conversation_answered_by_this_process_is_not_restored(self, server, folder):
        running = a_server(folder)
        saved_file(folder, running)
        running._seen.add(cache._conversation_id("Ada/thread-1"))

        assert running.restore("Ada/thread-1") is None
        assert server.requests == []

    def test_a_file_llama_server_refuses_is_removed(self, server, folder):
        server.restore_status = 400
        running = a_server(folder)
        path = saved_file(folder, running)

        assert running.restore("Ada/thread-1") is None
        assert not path.exists()

    def test_a_server_error_keeps_the_file(self, server, folder):
        server.restore_status = 500
        running = a_server(folder)
        path = saved_file(folder, running)

        assert running.restore("Ada/thread-1") is None
        assert path.exists()

    def test_a_file_of_another_server_is_never_offered(self, server, folder):
        writer = a_server(folder, identity="aaaaaaaaaaaa")
        saved_file(folder, writer)
        reader = a_server(folder, identity="bbbbbbbbbbbb")

        assert reader.restore("Ada/thread-1") is None
        assert server.requests == []

    def test_the_name_is_hashes_and_passes_llama_servers_filename_rule(self, folder):
        running = a_server(folder)
        name = running.file_name("Ada — a character/with a slash?")

        assert "Ada" not in name and "/" not in name and "?" not in name
        assert name.endswith(cache.SUFFIX)
        assert not name.endswith(".") and not name.startswith(" ")


# --------------------------------------------------------------------------- #
# Saving and the folder
# --------------------------------------------------------------------------- #


class TestTheSavedTokens:
    """The file's own token list, read as llama.cpp b10621 wrote it."""

    def test_a_packed_text_prompt_gives_its_tokens(self, tmp_path):
        path = slot_file(tmp_path / "a.slot", [2, 72, 105])

        assert cache.saved_tokens(path) == [2, 72, 105]

    def test_an_older_plain_list_gives_itself(self, tmp_path):
        path = slot_file(tmp_path / "a.slot", [2, 72, 105], plain=True)

        assert cache.saved_tokens(path) == [2, 72, 105]

    def test_a_file_with_a_picture_cannot_be_checked(self, tmp_path):
        path = slot_file(tmp_path / "a.slot", [2, 72, 105], pictures=1)

        assert cache.saved_tokens(path) is None

    def test_anything_else_is_not_read(self, tmp_path):
        for name, arguments in (("magic", dict(magic=0x12345678)),
                                ("version", dict(version=3)),
                                ("packing", dict(packed_version=2))):
            path = slot_file(tmp_path / f"{name}.slot", [2, 72, 105], **arguments)
            assert cache.saved_tokens(path) is None, name
        short = tmp_path / "short.slot"
        short.write_bytes(struct.pack("<III", cache.SEQUENCE_MAGIC, cache.SEQUENCE_VERSION, 50))
        assert cache.saved_tokens(short) is None
        assert cache.saved_tokens(tmp_path / "missing.slot") is None


class TestAWindowedModelIsReadBackOnlyWhenExact:
    """A sliding-window model's saved slot holds one window of positions, and
    llama.cpp resumes it exactly only where the new prompt extends it. So the
    file is read back only after the server's own rendering and tokenizing of
    the new request are shown to start with every saved token."""

    SAVED_TEXT = "<system>S<user>U1"

    def checking(self, folder):
        return cache.Server("http://127.0.0.1:58915", "k", folder, "abc123abc123", check=True)

    def body(self, *extra):
        messages = [{"role": "system", "content": "S"}, {"role": "user", "content": "U1"}]
        return {"messages": messages + list(extra), "stream": True}

    def saved(self, folder, running, text=None):
        path = folder / running.file_name("Ada/thread-1")
        return slot_file(path, tokenized(text or self.SAVED_TEXT, True))

    def test_a_prompt_that_extends_the_saved_one_is_read_back(self, server, folder):
        running = self.checking(folder)
        self.saved(folder, running)
        body = self.body({"role": "assistant", "content": "R"}, {"role": "user", "content": "U2"})

        assert running.restore("Ada/thread-1", body) == 0
        assert [url.rsplit("/", 1)[-1] for method, url, _, _ in server.requests] == [
            "apply-template", "tokenize", "slots", "0?action=restore"]
        tokenize = [sent for _, url, sent, _ in server.requests if url.endswith("/tokenize")][0]
        assert tokenize["add_special"] is True and tokenize["parse_special"] is True

    def test_a_prompt_that_parts_before_the_end_is_read_in_full(self, server, folder):
        running = self.checking(folder)
        self.saved(folder, running, "<system>S<user>U1<assistant>R")
        regenerated = self.body({"role": "assistant", "content": "Q"},
                                {"role": "user", "content": "U2"})

        assert running.restore("Ada/thread-1", regenerated) is None
        assert server.posted("restore") == []

    def test_the_same_prompt_again_is_read_in_full(self, server, folder):
        running = self.checking(folder)
        self.saved(folder, running)

        assert running.restore("Ada/thread-1", self.body()) is None
        assert server.posted("restore") == []

    def test_a_picture_in_the_request_is_read_in_full_without_asking(self, server, folder):
        running = self.checking(folder)
        self.saved(folder, running)
        pictured = self.body({"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}},
            {"type": "text", "text": "look"}]})

        assert running.restore("Ada/thread-1", pictured) is None
        assert server.requests == []

    def test_a_file_that_cannot_be_read_is_read_in_full(self, server, folder):
        running = self.checking(folder)
        (folder / running.file_name("Ada/thread-1")).write_bytes(b"not a slot file")

        assert running.restore("Ada/thread-1", self.body()) is None
        assert server.requests == []

    def test_a_model_without_a_window_is_not_checked(self, server, folder):
        running = a_server(folder)
        saved_file(folder, running)

        assert running.restore("Ada/thread-1", self.body()) == 0
        assert not any(url.endswith(("/apply-template", "/tokenize"))
                       for _, url, _, _ in server.requests)

    def test_the_reply_after_a_checked_restore_goes_to_that_slot(self, server, folder):
        running = self.checking(folder)
        client = cache.client("http://127.0.0.1:58915", "k", None, running)
        messages = [{"role": "system", "content": "S"}, {"role": "user", "content": "U1"},
                    {"role": "assistant", "content": "R"}, {"role": "user", "content": "U2"}]
        self.saved(folder, running)

        client.stream_conversation("Ada/thread-1", messages, 512, 7, lambda _text: None)

        assert server.chats()[-1]["id_slot"] == 0
        rendering = [sent for _, url, sent, _ in server.requests
                     if url.endswith("/apply-template")][0]
        assert rendering["messages"] == messages
        assert "id_slot" not in rendering and "verbose" not in rendering

    def test_a_start_of_a_windowed_model_checks_and_another_does_not(self, tmp_path):
        process = types.SimpleNamespace(port=58915, api_key="k")
        flags = [cache.SLOT_SAVE_FLAG, str(tmp_path)]

        assert cache.started(process, flags, "abc",
                             model=a_model(tmp_path, "w.gguf", window=1024)).check is True
        assert cache.started(process, flags, "abc", model=a_model(tmp_path)).check is False


class TestSaving:
    def test_the_slot_is_saved_under_the_conversations_name(self, server, folder):
        running = a_server(folder)

        assert running.save("Ada/thread-1", 3) is True
        assert server.posted("save") == [
            ("http://127.0.0.1:58915/slots/3?action=save",
             {"filename": running.file_name("Ada/thread-1")})]

    def test_turning_the_setting_off_stops_saving_at_once(self, server, folder, host):
        running = a_server(folder)
        host.shared.opts.set(cache.OPT_SAVE, False)

        assert running.save("Ada/thread-1", 3) is False
        assert server.posted("save") == []

    def test_a_refused_save_is_a_line_and_not_an_exception(self, server, folder):
        server.save_status = 500
        running = a_server(folder)

        assert running.save("Ada/thread-1", 3) is False

    def test_a_saved_conversation_is_not_restored_over_itself_later(self, server, folder):
        running = a_server(folder)
        saved_file(folder, running)

        running.save_later("Ada/thread-1", 3)
        assert wait_for(lambda: server.posted("save"))

        assert running.restore("Ada/thread-1") is None
        assert server.posted("restore") == []

    def test_each_save_is_followed_by_keeping_within_the_budget(self, server, folder, host):
        host.shared.opts.set(cache.OPT_BUDGET_GB, 1.5 / 1024)  # 1.5 MB
        running = a_server(folder)
        old = saved_file(folder, running, "Ada/old", size=1 * MB, age=7200)
        newer = saved_file(folder, running, "Ada/newer", size=1 * MB, age=60)
        mine = saved_file(folder, running, "Ada/thread-1", size=1 * MB)

        running.save("Ada/thread-1", 3)

        assert mine.exists()
        assert not old.exists() and not newer.exists()


class TestTheFolderBudget:
    def test_the_newest_are_kept_within_the_budget(self, folder):
        running = a_server(folder)
        paths = [saved_file(folder, running, f"Ada/{index}", size=MB, age=100 * (5 - index))
                 for index in range(5)]

        removed = cache.prune(folder, budget=int(2.5 * MB))

        assert removed == 3
        assert [path.exists() for path in paths] == [False, False, False, True, True]

    def test_the_file_just_written_stays_whatever_it_weighs(self, folder):
        running = a_server(folder)
        big = saved_file(folder, running, "Ada/big", size=3 * MB, age=500)
        other = saved_file(folder, running, "Ada/other", size=MB)

        cache.prune(folder, keep=big.name, budget=MB)

        assert big.exists()
        assert not other.exists()

    def test_other_files_in_the_folder_are_not_touched(self, folder):
        running = a_server(folder)
        newest = saved_file(folder, running, "Ada/1", size=MB)
        stray = folder / "notes.txt"
        stray.write_text("mine")
        stamp = time.time() - 3600
        os.utime(stray, (stamp, stamp))

        cache.prune(folder, budget=0)

        assert newest.exists()
        assert stray.exists()

    def test_usage_counts_the_saved_caches(self, folder):
        running = a_server(folder)
        saved_file(folder, running, "Ada/1", size=MB)
        saved_file(folder, running, "Ada/2", size=2 * MB)

        assert cache.usage() == (2, 3 * MB)


# --------------------------------------------------------------------------- #
# The reply
# --------------------------------------------------------------------------- #


def vendored_body(server, monkeypatch, sampling=None, **overrides):
    """What the vendored client sends for the same reply, captured on the wire."""
    from prompt_master.inference.llama_client import LlamaClient

    plain = LlamaClient("http://127.0.0.1:58915", "k", sampling)
    arguments = dict(messages=[{"role": "user", "content": "hi"}], max_tokens=512, seed=7,
                     on_text=lambda _text: None)
    arguments.update(overrides)
    plain.stream_chat(**arguments)
    return server.chats()[-1]


class TestTheReply:
    def saving(self, folder, sampling=None, **kwargs):
        return cache.client("http://127.0.0.1:58915", "k", sampling, a_server(folder, **kwargs))

    def test_the_body_is_the_vendored_one_with_the_slot_question_added(self, server,
                                                                        folder, monkeypatch):
        sampling = {"top_k": 64, "min_p": 0.05}
        expected = vendored_body(server, monkeypatch, sampling)
        client = self.saving(folder, sampling)

        client.stream_conversation("Ada/thread-1", [{"role": "user", "content": "hi"}], 512,
                                   7, lambda _text: None)
        sent = server.chats()[-1]

        added = {key: value for key, value in sent.items() if key not in expected}
        assert {key: sent[key] for key in expected} == expected
        assert added == {"verbose": True, "response_fields": ["id_slot"]}

    def test_the_words_arrive_as_they_do_through_the_vendored_client(self, server, folder):
        heard = []
        client = self.saving(folder)

        text = client.stream_conversation("Ada/thread-1", [{"role": "user", "content": "hi"}],
                                          512, 7, heard.append)

        assert heard == ["Hello", " there"]
        assert text == "Hello there"

    def test_the_slot_llama_server_names_is_the_one_saved(self, server, folder):
        server.answering = 4
        client = self.saving(folder)

        client.stream_conversation("Ada/thread-1", [{"role": "user", "content": "hi"}], 512,
                                   7, lambda _text: None)

        assert wait_for(lambda: server.saved)
        assert server.saved == [(4, client.saved_caches.file_name("Ada/thread-1"))]

    def test_the_first_reply_after_a_restore_is_sent_to_that_slot(self, server, folder):
        client = self.saving(folder)
        saved_file(folder, client.saved_caches)

        client.stream_conversation("Ada/thread-1", [{"role": "user", "content": "hi"}], 512,
                                   7, lambda _text: None)

        assert server.chats()[-1]["id_slot"] == 0
        assert wait_for(lambda: server.saved)
        assert server.saved[0][0] == 0

    def test_no_slot_is_named_without_a_restore(self, server, folder):
        client = self.saving(folder)

        client.stream_conversation("Ada/thread-1", [{"role": "user", "content": "hi"}], 512,
                                   7, lambda _text: None)

        assert "id_slot" not in server.chats()[-1]

    def test_a_stopped_reply_stops_its_words_and_is_not_saved(self, server, folder):
        stop = threading.Event()
        heard = []
        client = self.saving(folder)

        def hear(text):
            heard.append(text)
            stop.set()

        text = client.stream_conversation("Ada/thread-1", [{"role": "user", "content": "hi"}],
                                          512, 7, hear, stop)
        time.sleep(0.1)

        assert heard == ["Hello"] and text == "Hello"
        assert server.saved == []

    def test_a_reply_stopped_on_its_last_word_is_not_saved(self, server, folder):
        stop = threading.Event()
        client = self.saving(folder)

        def hear(text):
            if text == server.pieces[-1]:
                stop.set()

        client.stream_conversation("Ada/thread-1", [{"role": "user", "content": "hi"}], 512,
                                   7, hear, stop)
        time.sleep(0.1)

        assert server.saved == []

    def test_an_answer_that_names_no_slot_is_not_saved(self, server, folder, monkeypatch):
        original = server.chat

        def quiet(body):
            return [line for line in original(dict(body, verbose=False))]

        monkeypatch.setattr(server, "chat", quiet)
        client = self.saving(folder)

        client.stream_conversation("Ada/thread-1", [{"role": "user", "content": "hi"}], 512,
                                   7, lambda _text: None)
        time.sleep(0.1)

        assert server.saved == []

    def test_saving_off_is_the_vendored_request_exactly(self, server, folder, monkeypatch,
                                                        host):
        expected = vendored_body(server, monkeypatch)
        host.shared.opts.set(cache.OPT_SAVE, False)
        client = self.saving(folder)

        client.stream_conversation("Ada/thread-1", [{"role": "user", "content": "hi"}], 512,
                                   7, lambda _text: None)

        assert server.chats()[-1] == expected
        assert server.posted("restore") == [] and server.saved == []

    def test_every_other_caller_keeps_the_vendored_method(self, server, folder, monkeypatch):
        expected = vendored_body(server, monkeypatch)
        client = self.saving(folder)

        client.stream_chat([{"role": "user", "content": "hi"}], 512, 7, lambda _text: None)

        assert server.chats()[-1] == expected


class TestReadingTheSlot:
    def test_the_slot_is_read_off_the_verbose_field(self):
        found = {}
        lines = ['data: {"choices": [], "__verbose": {"id_slot": 5}}', "data: [DONE]"]

        assert list(cache.watching(lines, found)) == lines
        assert found == {"slot": 5}

    def test_nothing_that_is_not_a_slot_number_counts(self):
        for line in ('data: {"__verbose": {"id_slot": true}}',
                     'data: {"__verbose": {"id_slot": -1}}',
                     'data: {"__verbose": {"id_slot": "3"}}',
                     'data: {"__verbose": "id_slot 3"}',
                     'data: {not json "__verbose"',
                     ': comment "__verbose": {"id_slot": 2}'):
            found = {}
            list(cache.watching([line], found))
            assert found == {}, line

    def test_the_fields_ask_for_the_slot_alone(self):
        assert cache.slot_fields(None) == {"verbose": True, "response_fields": ["id_slot"]}
        assert cache.slot_fields(2)["id_slot"] == 2


# --------------------------------------------------------------------------- #
# The runtime and the conversation path
# --------------------------------------------------------------------------- #


class TestTheRuntimeHandsItOut:
    def runtime_with(self, saving):
        found = runtime.Runtime()
        found._process = types.SimpleNamespace(port=58915, api_key="k")
        found._saved_caches = saving
        return found

    def test_a_server_started_able_to_save_hands_out_the_saving_client(self, folder):
        running = a_server(folder)
        found = self.runtime_with(running)
        configuration = types.SimpleNamespace(profile=None)

        made = found._client(configuration)

        assert callable(getattr(made, "stream_conversation", None))
        assert made.saved_caches is running

    def test_one_that_was_not_hands_out_the_vendored_client(self):
        from prompt_master.inference.llama_client import LlamaClient

        found = self.runtime_with(None)

        made = found._client(types.SimpleNamespace(profile=None))

        assert type(made) is LlamaClient

    def test_stopping_the_server_forgets_it(self, folder):
        found = self.runtime_with(a_server(folder))
        found._process = types.SimpleNamespace(port=58915, api_key="k", stop=lambda: None)

        found._stop_locked("stop requested")

        assert found._saved_caches is None


class TestTheConversationPath:
    """A reply that names its conversation uses the saving method when its
    client has one; everything else streams exactly as before."""

    class Both:
        def __init__(self):
            self.saving, self.plain = [], []

        def stream_conversation(self, conversation, messages, max_tokens, seed, on_text,
                                cancel=None, temperature=0.85, top_p=0.95):
            self.saving.append(conversation)
            on_text("ok")
            return "ok"

        def stream_chat(self, messages, max_tokens, seed, on_text, cancel=None,
                        temperature=0.85, top_p=0.95):
            self.plain.append(messages)
            on_text("ok")
            return "ok"

    @pytest.fixture
    def installed(self, monkeypatch, host):
        import mc_broker

        mc_broker.clear()
        monkeypatch.setattr(mc_broker, "host_busy", lambda: False)
        client = self.Both()
        monkeypatch.setattr(sessions, "_client",
                            lambda needs_vision=False, reserve=0, role='', cancel=None: client)
        monkeypatch.setattr(sessions, "_placement_notes", lambda role="": [])
        yield client
        mc_broker.clear()

    def test_a_named_conversation_is_saved(self, installed):
        request = sessions.ChatRequest(messages=[{"role": "user", "content": "hi"}],
                                       conversation="Ada/thread-1")

        events = list(sessions.conversation(request, sessions.Cancellation()))

        assert installed.saving == ["Ada/thread-1"] and installed.plain == []
        assert events[-1].kind == sessions.DONE

    def test_an_unnamed_request_streams_as_before(self, installed):
        request = sessions.ChatRequest(messages=[{"role": "user", "content": "hi"}])

        list(sessions.conversation(request, sessions.Cancellation()))

        assert installed.saving == [] and len(installed.plain) == 1

    def test_the_reply_operation_names_its_conversation(self, monkeypatch, tmp_path, host):
        import mc_llm_conversation_feed as feed
        import mc_llm_conversation_ops as ops
        import mc_llm_conversation_service as service
        import mc_llm_conversation_store as store_module
        from prompt_master.chat.characters import Character, CharacterStore
        from prompt_master.chat.history import ASSISTANT, USER, ChatStore
        from test_conversation_service import envelope

        monkeypatch.setattr(mc_llm_paths, "data_root", lambda: tmp_path)
        service.reset()
        store_module.registry.reset()
        ops.reset()
        feed.reset()
        try:
            CharacterStore(tmp_path / "characters").save(Character(name="Ada"))
            chats = ChatStore(tmp_path / "chats")
            thread = chats.new("Ada")
            thread.append(USER, "ask")
            thread.append(ASSISTANT, "reply")
            chats.save(thread)
            asked = []
            events = [sessions.Event(sessions.CHUNK, "ok"), sessions.Event(sessions.DONE, "ok")]
            monkeypatch.setattr(sessions, "conversation",
                                lambda request, cancel: (asked.append(request), iter(events))[1])

            outcome = service.submit(envelope("send", thread, payload={"text": "and now?"},
                                              operation_id="op-named"))
            assert ops.drain(timeout=10)

            assert outcome["ok"] is True, outcome
            assert asked[-1].conversation == f"Ada/{thread.identifier}"
        finally:
            ops.reset()
            service.reset()
            store_module.registry.reset()
            feed.reset()


class TestTheSettingsPageOffersIt:
    def test_both_settings_are_registered_with_their_defaults(self):
        import pathlib

        root = pathlib.Path(__file__).resolve().parent.parent
        source = (root / "scripts" / "model_chain.py").read_text(encoding="utf-8")

        assert "mc_llm_slot_cache.OPT_SAVE: shared.OptionInfo(\n                True," in source
        assert ("mc_llm_slot_cache.OPT_BUDGET_GB: shared.OptionInfo(\n"
                "                mc_llm_slot_cache.DEFAULT_BUDGET_GB,") in source

    def test_saving_is_on_and_the_budget_is_a_working_set_by_default(self, host):
        assert cache.enabled() is True
        assert cache.budget_bytes() == int(cache.DEFAULT_BUDGET_GB * GB)
        assert 1 <= cache.DEFAULT_BUDGET_GB <= 16
