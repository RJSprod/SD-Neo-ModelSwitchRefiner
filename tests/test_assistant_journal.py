"""The Forge Assistant's journal: what the page saw, in the extension's log.

Built for the drag's account, at the user's word after it had gone to the
browser's console ("i meant our log file ... model_chain.log ... write it
here, with other logging content"): one line of that log per entry, through
the `model_chain` logger like every other line, as the page sent it. The
drag's account was removed at the user's word too ("i should not see those
drag events anymore"); the quiet-page watch's churn line is what it keeps,
and a drag entry from a page loaded before the update is answered and never
written.
"""

from __future__ import annotations

import inspect
import json
import logging
import re

import pytest

import mc_assistant_journal as journal
import mc_llm_conversation_api as api
import mc_llm_paths
import mc_logfile

@pytest.fixture
def lines():
    """The `model_chain` logger's records, as its own handlers see them -- it
    does not hand them up to the root, so pytest's caplog never sees them."""
    logger = logging.getLogger("model_chain")
    kept = []

    class Keep(logging.Handler):
        def emit(self, record):
            kept.append(record)

    handler = Keep(level=logging.INFO)
    was = logger.level
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    try:
        yield kept
    finally:
        logger.removeHandler(handler)
        logger.setLevel(was)


def unstamped(message: str) -> str:
    """The logger's own filter puts the clock after "Model Chain"; the line
    the module wrote is the one without it."""
    return re.sub(r"^Model Chain \[[^\]]+\]: ", "Model Chain: ", message)


ENTRY = {"kind": "churn", "runs": 24, "seconds": 7.3, "quiet": 12,
         "targets": [{"node": "div#txt2img_settings", "count": 40},
                     {"node": "div.progress", "count": 12}]}

# What a page loaded before the drag's logs were removed still sends after
# every drag, until it is reloaded.
DRAG = {"kind": "drag", "pointer": "mouse", "handle": "launcher", "travel": 180.4,
        "ended": "pointerup", "astray": ["the page scrolled by 0, 240"],
        "trace": [{"t": 0, "e": "pointerdown"}, {"t": 90, "e": "pointerup"}],
        "environment": {"userAgent": "LibreWolf"}}


class TestTheLine:
    def test_an_entry_is_one_line_that_reads_before_its_json(self):
        found = journal.line(ENTRY)

        head, _, tail = found.partition("): ")
        assert head.startswith("Model Chain: Forge Assistant churn (24 after-update runs")
        assert json.loads(tail) == {key: value for key, value in ENTRY.items() if key != "kind"}
        assert "\n" not in found, "one line, so the file reads one entry per line"

    def test_a_page_that_never_settles_is_summed_up_by_what_kept_changing(self):
        """The quiet-page watch's line: how often the page updated while
        nobody was doing anything, and the nodes whose child lists changed
        most, which is where the callback that keeps it updating is found."""
        found = journal.line({"kind": "churn", "runs": 24, "seconds": 7.3, "quiet": 12,
                              "targets": [{"node": "div#txt2img_settings", "count": 40},
                                          {"node": "div.progress", "count": 12}]})

        head, _, tail = found.partition("): ")
        assert head == ("Model Chain: Forge Assistant churn (24 after-update runs in 7.3 s, "
                        "nothing pressed for 12 s; most changed: div#txt2img_settings 40, "
                        "div.progress 12")
        assert json.loads(tail)["runs"] == 24
        assert journal.summary({"kind": "churn"}) == "? after-update runs"

    def test_the_journal_keeps_churn_entries_and_no_others(self):
        """An entry says what it is, and churn is all the journal keeps: no
        kind assumed for an entry without one (it was a drag once), and none
        it does not know -- a drag's line is never made, not even here."""
        for bad in ({}, {"pointer": "pen"}, {"kind": "drag"}, {"kind": "no spaces"},
                    {"kind": "Churn"}, {"kind": 7}, {"kind": None}, {"kind": ["churn"]}):
            with pytest.raises(ValueError, match="one kind of entry: churn"):
                journal.line(bad)

    def test_what_is_not_an_entry_is_refused(self):
        with pytest.raises(ValueError):
            journal.line(["churn"])
        with pytest.raises(ValueError):
            journal.line({"kind": "churn", "targets": {1, 2}})
        with pytest.raises(ValueError):
            journal.line({"kind": "churn", "targets": "x" * (journal.MAX_ENTRY + 1)})


class TestTheLog:
    def test_an_entry_goes_through_the_extensions_logger(self, lines):
        assert journal.record(ENTRY) == "model_chain.log"

        assert len(lines) == 1 and lines[0].levelno == logging.INFO
        assert unstamped(lines[0].getMessage()) == journal.line(ENTRY)

    def test_a_drag_from_a_page_loaded_before_the_update_is_never_written(self, lines):
        """"remove those flyout movement logs from all logs, i should not see
        those drag events anymore": the page sends none, and a page loaded
        before the update still does after every drag until it is reloaded.
        Answered, and not a line anywhere -- the logger hears nothing."""
        assert journal.record(dict(DRAG)) is None
        assert journal.record({"kind": "drag"}) is None

        assert lines == []

    def test_the_line_lands_in_model_chain_log_between_other_lines(self, tmp_path, monkeypatch,
                                                                    lines):
        """End to end: the file `mc_logfile` writes, with other logging around it."""
        monkeypatch.setattr(mc_llm_paths, "data_root", lambda: tmp_path)
        assert mc_logfile.attached() is None, "no handler from another test"
        assert mc_logfile.attach() is True
        try:
            logger = logging.getLogger("model_chain")
            logger.info("Model Chain: something before")
            journal.record(DRAG)
            journal.record(ENTRY)
            logger.info("Model Chain: something after")
            text = mc_logfile.path().read_text(encoding="utf-8")
        finally:
            handler = mc_logfile.attached()
            logger.removeHandler(handler)
            handler.close()

        written = text.splitlines()
        assert mc_logfile.path() == tmp_path / "logs" / "model_chain.log"
        assert written[-3].endswith(": something before")
        assert unstamped(written[-2].split(" ", 1)[1]) == journal.line(ENTRY), written[-2]
        assert written[-1].endswith(": something after")
        assert "Forge Assistant drag" not in text, "the drag went nowhere"


class TestTheRoute:
    def test_the_route_is_registered_with_the_others(self):
        class App:
            added = {}

            def add_api_route(self, path, handler, methods):
                self.added[path] = (handler, list(methods))

        assert api.JOURNAL_ROUTE == "/model-chain/conversation/v2/journal"
        assert api.JOURNAL_ROUTE in api.ROUTES
        if api.Request is None:
            pytest.skip("FastAPI is not installed")
        app = App()
        api.install(app=app)

        handler, methods = app.added[api.JOURNAL_ROUTE]
        assert methods == ["POST"]
        assert inspect.iscoroutinefunction(handler)

    def test_the_handler_answers_with_the_files_name(self, lines):
        payload, status = api.journal(dict(ENTRY))

        assert (payload, status) == ({"ok": True, "file": "model_chain.log"}, 200)
        assert [unstamped(record.getMessage()) for record in lines] == [journal.line(ENTRY)]

    def test_a_stale_pages_drag_is_answered_and_not_written(self, lines):
        """A refusal would put a line on that page's console after every
        drag; an answer that names no file puts nothing anywhere."""
        payload, status = api.journal(dict(DRAG))

        assert (payload, status) == ({"ok": True, "file": None}, 200)
        assert lines == []

    def test_what_is_not_an_entry_is_refused_with_its_reason(self, lines):
        payload, status = api.journal({"kind": "no spaces"})

        assert status == 400 and payload["ok"] is False
        assert payload["error"]["code"] == "INVALID_INPUT"
        assert payload["error"]["message"] == "The journal keeps one kind of entry: churn."
        assert not any("Forge Assistant" in record.getMessage() for record in lines)
