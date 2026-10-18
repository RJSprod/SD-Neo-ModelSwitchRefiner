"""No path on this machine and no file of the user's in anything this extension logs.

The user's rule, now a standing convention (CLAUDE.md): a log line never names
a path on this machine, a file or folder the user made or chose, or the user's
own text. Two layers hold it, and both are tested here: the lines that used to
carry one were rewritten, and a filter on the ``model_chain`` logger redacts
any absolute path that reaches a record anyway -- in the message, in an
argument, in a traceback -- so a line written tomorrow cannot leak one.
"""

from __future__ import annotations

import io
import logging
import pathlib
import re
import types

import pytest

import mc_log_privacy
import mc_memory

P = mc_log_privacy.PLACEHOLDER


class TestRedaction:
    @pytest.mark.parametrize("text, expected", [
        (r"written to C:\Roots\Neo3\model_chain_llm\logs\model_chain.log", f"written to {P}"),
        (r"saved to C:\Users\Me\Pictures\out 2.png, done", f"saved to {P} 2.png, done"),
        ("found at D:/models/krea/x.safetensors.", f"found at {P}."),
        (r"copied from \\nas\share\renders\a.wav", f"copied from {P}"),
        ("the attachment at /home/me/Pictures/cat.png could not be read",
         f"the attachment at {P} could not be read"),
        ("under /Users/me/Library/x (kept)", f"under {P} (kept)"),
        ("mounted at /mnt/d/forge/outputs/", f"mounted at {P}"),
        ("the folder ~/Desktop/renders holds it", f"the folder {P} holds it"),
        ("see file:///C:/x/y.txt now", f"see {P} now"),
        ("the root /tmp is used", f"the root {P} is used"),
    ])
    def test_a_path_is_replaced_and_the_sentence_kept(self, text, expected):
        assert mc_log_privacy.redact(text) == expected

    @pytest.mark.parametrize("text", [
        "Voice Chat refused a request to /model-chain/voice-box/outputs — no page token",
        "routes registered at /model-chain/voice/status, /model-chain/voice/stt",
        "Model Chain [13:35:43.484]: this log is also being written",
        r"Krea\kroma-v0.3.1-turbo-opd-W8A8.safetensors is in VRAM — 18.4 GB resident",
        "page over https://localhost:7868/ answered",
        "the model/homework split",
        "ratio 3/4 of the card, a/b",
        "a 2x16x1x128x128 latent",
    ])
    def test_what_is_not_a_path_is_left_alone(self, text):
        assert mc_log_privacy.redact(text) == text

    def test_empty_is_empty(self):
        assert mc_log_privacy.redact("") == ""

    def test_a_traceback_loses_its_source_paths(self):
        trace = ('Traceback (most recent call last):\n'
                 '  File "C:\\Roots\\Neo3\\extensions\\x\\mc_gallery_quadrant.py", line 334, in save\n'
                 '    written = save_copy(source, folder)\n'
                 "PermissionError: [Errno 13] Permission denied: 'C:\\\\Users\\\\Me\\\\out.png'")
        cleaned = mc_log_privacy.redact(trace)
        assert "Roots" not in cleaned and "Users" not in cleaned
        assert f'File "{P}", line 334, in save' in cleaned


@pytest.fixture
def sink():
    """A handler on the model_chain logger, as the file handler is."""
    logger = logging.getLogger("model_chain")
    buffer = io.StringIO()
    handler = logging.StreamHandler(buffer)
    handler.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
    logger.addHandler(handler)
    try:
        yield buffer
    finally:
        logger.removeHandler(handler)


class TestTheFilter:
    def test_it_is_on_the_logger_the_extension_logs_through(self):
        logger = mc_memory._make_logger()
        assert mc_log_privacy.installed(logger)
        assert logger is logging.getLogger("model_chain")

    def test_it_is_installed_once(self):
        logger = mc_memory._make_logger()
        mc_memory._make_logger()
        assert sum(isinstance(f, mc_log_privacy.NoPaths) for f in logger.filters) == 1

    def test_a_path_in_the_message_never_reaches_a_handler(self, sink):
        logging.getLogger("model_chain").info("Model Chain: saved to C:\\Users\\Me\\a.png")
        assert "Users" not in sink.getvalue()
        assert f"saved to {P}" in sink.getvalue()

    def test_a_path_in_an_argument_never_reaches_a_handler(self, sink):
        logging.getLogger("model_chain").info("Model Chain: could not read %s", "/home/me/x.json")
        assert "/home" not in sink.getvalue()
        assert f"could not read {P}" in sink.getvalue()

    def test_a_traceback_never_reaches_a_handler_with_its_paths(self, sink):
        try:
            raise OSError("Permission denied: '/home/me/out.png'")
        except OSError:
            logging.getLogger("model_chain").warning("Model Chain: a save failed", exc_info=True)
        written = sink.getvalue()
        assert "Traceback" in written and "OSError" in written
        assert "/home/" not in written
        assert str(pathlib.Path(__file__).parent) not in written, "the test file's own path too"

    def test_the_stamped_line_is_the_redacted_one(self, sink):
        logging.getLogger("model_chain").info("Model Chain: see C:\\x\\y.log")
        assert re.search(r"Model Chain \[\d\d:\d\d:\d\d\.\d{3}\]: see <path>", sink.getvalue())

    def test_a_record_it_cannot_read_still_passes(self):
        record = types.SimpleNamespace(getMessage=lambda: (_ for _ in ()).throw(ValueError()),
                                       args=(), exc_info=None, exc_text=None, stack_info=None)
        assert mc_log_privacy.NoPaths().filter(record) is True


class TestTheLinesThatUsedToCarryOne:
    """Each used to say a path or a user's file name; none does now, so none
    needs the placeholder either."""

    def test_the_log_file_is_named_without_its_folder(self, tmp_path, monkeypatch, sink):
        import mc_logfile

        monkeypatch.setattr(mc_logfile, "directory", lambda: tmp_path / "logs")
        for handler in list(mc_logfile.logger.handlers):
            if getattr(handler, mc_logfile._MARK, False):
                mc_logfile.logger.removeHandler(handler)
        try:
            assert mc_logfile.attach() is True
        finally:
            for handler in list(mc_logfile.logger.handlers):
                if getattr(handler, mc_logfile._MARK, False):
                    mc_logfile.logger.removeHandler(handler)
                    handler.close()
        [line] = [l for l in sink.getvalue().splitlines() if "also being written" in l]
        assert line.endswith("also being written to model_chain.log in the logs folder of the "
                             "LLM data directory")
        assert str(tmp_path) not in line and P not in line

    def test_the_voice_host_line_has_no_root(self, host):
        import mc_voice_models

        line = mc_voice_models.describe_host()
        assert "voice root" not in line
        assert P not in mc_log_privacy.redact(line)

    def test_a_gallery_quadrant_save_logs_nothing(self, tmp_path, monkeypatch, sink):
        import mc_gallery_quadrant as gq

        source = tmp_path / "source" / "picture.png"
        source.parent.mkdir()
        source.write_bytes(b"png")
        folder = tmp_path / "saves"
        folder.mkdir()
        monkeypatch.setattr(gq, "destination_of", lambda opts: gq.DESTINATION_FOLDER)
        monkeypatch.setattr(gq, "resolve_folder", lambda value, data_dir=None: folder)
        monkeypatch.setattr(gq, "resolve_source", lambda src, roots: source)

        answer = gq.save({"src": "file=picture.png"}, opts=types.SimpleNamespace(),
                         roots=[source.parent], data_dir=tmp_path)

        assert answer["ok"] is True
        assert "picture" not in sink.getvalue() and "saves" not in sink.getvalue()
        assert "Gallery Quadrant" not in sink.getvalue()

    @pytest.mark.parametrize("module, gone", [
        ("mc_gallery_quadrant", "Gallery Quadrant saved %s to %s"),
        ("mc_logfile", "also being written to %s\""),
        ("mc_voice_api", "data directory is %s"),
        ("mc_voice_models", "voice root {"),
        ("mc_voice_box", "saved “%s” to %s"),
        ("mc_llm_runtime", "llama-server log — %s"),
        ("mc_llm_chat_panel", "attachments into %s"),
    ])
    def test_the_old_wording_is_gone(self, module, gone):
        source = (pathlib.Path(mc_log_privacy.__file__).parent / f"{module}.py").read_text(
            encoding="utf-8")
        assert gone not in source

    def test_every_module_logs_through_the_filtered_logger(self):
        """A logger of another name has neither the filter nor the file."""
        root = pathlib.Path(mc_log_privacy.__file__).parent
        strays = []
        for path in sorted(root.glob("mc_*.py")) + sorted((root / "scripts").glob("*.py")):
            text = path.read_text(encoding="utf-8")
            for found in re.findall(r"logging\.getLogger\(([^)]*)\)", text):
                if found.strip() != '"model_chain"':
                    strays.append(f"{path.name}: getLogger({found})")
        assert strays == []
