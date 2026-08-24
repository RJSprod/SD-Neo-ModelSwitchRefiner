"""The extension's log, in a file.

Everything here logs through ``logging.getLogger("model_chain")``, and until
this module that logger had exactly one handler: the host's console. A line
about why a switch was slow, or about what the page did with the Literal Prompt
boxes, existed only in the terminal Forge was started from -- which is no help
at all to somebody running it as a service, or reading it the next morning.

So a second handler, writing the same lines to a file in the folder this
extension already keeps things in, beside the ``llama-server.log`` the managed
runtime writes.

What the tests are about is the two ways a second handler goes wrong: attaching
twice, which writes every line twice, and attaching to a place that cannot be
written, which must leave the console log exactly as it was.
"""

from __future__ import annotations

import logging
import logging.handlers

import pytest

import mc_llm_paths
import mc_logfile


@pytest.fixture
def detached():
    """The logger with this module's handler off it, before and after."""
    def strip():
        for handler in list(mc_logfile.logger.handlers):
            if getattr(handler, mc_logfile._MARK, False):
                mc_logfile.logger.removeHandler(handler)
                handler.close()

    strip()
    yield
    strip()


@pytest.fixture
def root(tmp_path, monkeypatch):
    """An LLM data root of our own, the way a user's setting points at one."""
    monkeypatch.setattr(mc_llm_paths, "data_root", lambda: tmp_path)
    return tmp_path


class TestWhereItWrites:
    def test_beside_the_llms_own_log(self, root, detached):
        assert mc_logfile.path() == root / "logs" / mc_logfile.FILENAME

    def test_it_follows_the_configured_root(self, tmp_path, monkeypatch, detached):
        """`model_chain_llm_root` can move the whole folder, which is the reason
        this waits for the settings to load before opening anything."""
        moved = tmp_path / "somewhere else"
        monkeypatch.setattr(mc_llm_paths, "data_root", lambda: moved)

        assert mc_logfile.path() == moved / "logs" / mc_logfile.FILENAME

    def test_a_root_that_cannot_be_worked_out_is_not_a_crash(self, monkeypatch,
                                                             detached):
        def raise_it():
            raise RuntimeError("no data directory")

        monkeypatch.setattr(mc_llm_paths, "data_root", raise_it)

        assert mc_logfile.path() is None
        assert mc_logfile.attach() is False


class TestAttaching:
    def test_the_lines_reach_the_file(self, root, detached):
        assert mc_logfile.attach() is True

        mc_logfile.logger.warning("Model Chain: a thing worth keeping")

        written = (root / "logs" / mc_logfile.FILENAME).read_text(encoding="utf-8")
        assert "a thing worth keeping" in written
        assert "WARNING" in written

    def test_the_first_line_names_the_file(self, root, detached):
        """So that "where did my logs go" answers itself from either end."""
        mc_logfile.attach()

        written = (root / "logs" / mc_logfile.FILENAME).read_text(encoding="utf-8")
        assert str(root / "logs" / mc_logfile.FILENAME) in written

    def test_attaching_twice_leaves_one_handler(self, root, detached):
        """`on_app_started` can fire more than once across a UI reload, and a
        second handler on the same file is every line written twice."""
        mc_logfile.attach()
        mc_logfile.attach()
        mc_logfile.attach()

        ours = [h for h in mc_logfile.logger.handlers
                if getattr(h, mc_logfile._MARK, False)]
        assert len(ours) == 1

    def test_the_console_handler_is_left_alone(self, root, detached):
        """A second handler, not a replacement: the terminal keeps working."""
        before = [h for h in mc_logfile.logger.handlers
                  if not getattr(h, mc_logfile._MARK, False)]
        mc_logfile.attach()
        after = [h for h in mc_logfile.logger.handlers
                 if not getattr(h, mc_logfile._MARK, False)]

        assert before == after

    def test_it_rotates_rather_than_growing_forever(self, root, detached):
        mc_logfile.attach()

        handler = mc_logfile.attached()
        assert isinstance(handler, logging.handlers.RotatingFileHandler)
        assert handler.maxBytes == mc_logfile.MAX_BYTES
        assert handler.backupCount == mc_logfile.BACKUPS

    def test_a_folder_that_cannot_be_made_is_not_a_crash(self, root, monkeypatch,
                                                         detached):
        """A read-only data directory. The console log carries on."""
        def refuse(*args, **kwargs):
            raise PermissionError("read-only")

        monkeypatch.setattr(type(root), "mkdir", refuse, raising=False)

        assert mc_logfile.attach() is False
        assert mc_logfile.attached() is None
        mc_logfile.logger.info("Model Chain: still logging to the console")


class TestItIsRegistered:
    def test_the_extension_asks_for_it_at_app_start(self):
        """Both halves of this: nothing writes to the file until the settings
        that say where it goes are loaded."""
        import model_chain  # noqa: F401  -- imported for its registrations
        from modules import script_callbacks

        assert mc_logfile.attach in script_callbacks.registered["app_started"]

    def test_the_literal_prompt_report_goes_in_it_too(self, root, detached,
                                                      caplog):
        """The question that started this: what the page did with those two
        boxes, in the log rather than in the developer tools."""
        import mc_literal_report

        mc_logfile.attach()
        mc_literal_report.note({"boxesFound": True, "claimed": False,
                                "autocompleteInstalled": True, "config": "loaded",
                                "thirdPartyBoxes": False})

        written = (root / "logs" / mc_logfile.FILENAME).read_text(encoding="utf-8")
        assert "has not claimed the Literal Prompt boxes" in written
        assert "third party textboxes" in written
