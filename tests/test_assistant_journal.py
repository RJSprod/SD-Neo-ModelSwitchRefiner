"""The Forge Assistant's journal: a drag's account into the extension's log.

The user's words, after the account had gone to the browser's console: "i
meant our log file ... model_chain.log ... write it here, with other logging
content". One line of that log per account, through the `model_chain` logger
like every other line, as the page sent it.
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


ACCOUNT = {"kind": "drag", "pointer": "mouse", "handle": "launcher", "travel": 180.4,
           "ended": "pointerup", "astray": ["the page scrolled by 0, 240"],
           "trace": [{"t": 0, "e": "pointerdown"}, {"t": 90, "e": "pointerup"}],
           "environment": {"userAgent": "LibreWolf"}}


class TestTheLine:
    def test_an_account_is_one_line_that_reads_before_its_json(self):
        found = journal.line(ACCOUNT)

        head, _, tail = found.partition("): ")
        assert head == ("Model Chain: Forge Assistant drag (mouse on the launcher, 180 px, "
                        "ended by pointerup, the page scrolled by 0, 240")
        assert json.loads(tail) == {key: value for key, value in ACCOUNT.items() if key != "kind"}
        assert "\n" not in found, "one line, so the file reads one account per line"

    def test_a_drag_that_moved_the_handle_alone_says_so(self):
        found = journal.summary({"pointer": "touch", "handle": "panel", "ended": "pointerup",
                                 "astray": []})

        assert found == "touch on the panel, ended by pointerup, nothing else moved"

    def test_the_kind_defaults_to_a_drag_and_is_one_word(self):
        assert journal.line({"pointer": "pen"}).startswith("Model Chain: Forge Assistant drag (")
        for bad in ("no spaces", "", "Drag", 7, None):
            with pytest.raises(ValueError):
                journal.line({"kind": bad})

    def test_what_is_not_an_account_is_refused(self):
        with pytest.raises(ValueError):
            journal.line(["drag"])
        with pytest.raises(ValueError):
            journal.line({"kind": "drag", "trace": {1, 2}})
        with pytest.raises(ValueError):
            journal.line({"kind": "drag", "trace": "x" * (journal.MAX_ENTRY + 1)})


class TestTheLog:
    def test_an_account_goes_through_the_extensions_logger(self, lines):
        assert journal.record(ACCOUNT) == "model_chain.log"

        assert len(lines) == 1 and lines[0].levelno == logging.INFO
        assert unstamped(lines[0].getMessage()) == journal.line(ACCOUNT)

    def test_the_line_lands_in_model_chain_log_between_other_lines(self, tmp_path, monkeypatch,
                                                                    lines):
        """End to end: the file `mc_logfile` writes, with other logging around it."""
        monkeypatch.setattr(mc_llm_paths, "data_root", lambda: tmp_path)
        assert mc_logfile.attached() is None, "no handler from another test"
        assert mc_logfile.attach() is True
        try:
            logger = logging.getLogger("model_chain")
            logger.info("Model Chain: something before")
            journal.record(ACCOUNT)
            logger.info("Model Chain: something after")
            text = mc_logfile.path().read_text(encoding="utf-8")
        finally:
            handler = mc_logfile.attached()
            logger.removeHandler(handler)
            handler.close()

        written = text.splitlines()
        assert mc_logfile.path() == tmp_path / "logs" / "model_chain.log"
        assert written[-3].endswith(": something before")
        assert unstamped(written[-2].split(" ", 1)[1]) == journal.line(ACCOUNT), written[-2]
        assert written[-1].endswith(": something after")


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
        payload, status = api.journal(dict(ACCOUNT))

        assert (payload, status) == ({"ok": True, "file": "model_chain.log"}, 200)
        assert [unstamped(record.getMessage()) for record in lines] == [journal.line(ACCOUNT)]

    def test_what_is_not_an_account_is_refused_with_its_reason(self, lines):
        payload, status = api.journal({"kind": "no spaces"})

        assert status == 400 and payload["ok"] is False
        assert payload["error"]["code"] == "INVALID_INPUT"
        assert payload["error"]["message"] == "An account says what kind it is, in one word."
        assert not any("Forge Assistant" in record.getMessage() for record in lines)
