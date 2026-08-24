"""The line the browser's report becomes in the log.

Two extensions have to cooperate for the Literal Prompt boxes to complete tags
and to receive a LoRA card, and Python can see neither of them. This is the
other end of ``javascript/model_chain_literals.js``'s report: it turns nine
booleans into one sentence that says what is wrong, in the log the rest of this
extension writes to, so that "it does not work here" has an answer that does not
begin with "open the developer tools".

The tests that matter are the ones about what it *cannot* do. It is handed a
payload from a browser, so it treats every value as untrusted: a fixed set of
keys, each coerced, and everything else ignored. There is nowhere for a prompt
to get in.
"""

from __future__ import annotations

import logging

import pytest

import mc_literal_report


def report(**overrides) -> dict:
    """A page where everything worked, unless a test says otherwise."""
    found = {"boxesFound": True, "claimed": True, "autocompleteInstalled": True,
             "listWrapped": True, "inTheirList": True, "config": "loaded",
             "thirdPartyBoxes": True, "promptFamily": True, "placed": True}
    found.update(overrides)
    return found


class TestWhatItSays:
    def test_a_working_page_is_not_a_warning(self):
        line, wrong = mc_literal_report.describe(report())

        assert wrong is False
        assert "have tag completion" in line

    def test_tag_completion_working_without_the_family_is_still_wrong(self):
        """The two are separate promises: one is Tag Autocomplete's, the other
        is the Extra Networks browser's. A page can keep one and lose the
        other, and the line has to say which."""
        line, wrong = mc_literal_report.describe(report(promptFamily=False))

        assert wrong is True
        assert "was not" in line

    def test_no_tag_autocomplete_is_reported_but_is_not_a_fault(self):
        """Not installing an extension is not a bug in this one."""
        line, wrong = mc_literal_report.describe(
            report(claimed=False, autocompleteInstalled=False, config="missing",
                   thirdPartyBoxes=None, listWrapped=False, inTheirList=False))

        assert wrong is False
        assert "no Tag Autocomplete on this page" in line

    def test_the_third_party_setting_is_named_when_it_is_off(self):
        """The one cause the user can fix themselves, and the one that needs a
        full restart rather than a UI reload -- so the line says so."""
        line, wrong = mc_literal_report.describe(
            report(claimed=False, thirdPartyBoxes=False))

        assert wrong is True
        assert "third party textboxes" in line
        assert "restart" in line

    def test_a_config_that_has_not_loaded_says_so(self):
        line, _ = mc_literal_report.describe(report(claimed=False, config="null"))

        assert "still loading" in line

    def test_a_list_that_could_not_be_extended_says_so(self):
        line, _ = mc_literal_report.describe(
            report(claimed=False, listWrapped=False, inTheirList=False))

        assert "could not extend the list" in line

    def test_boxes_that_are_not_on_the_page_are_the_first_thing_said(self):
        line, wrong = mc_literal_report.describe(report(boxesFound=False))

        assert wrong is True
        assert "without finding themselves" in line


class TestWhatItCannotBeMadeToDo:
    """It is handed a payload from a browser."""

    def test_no_value_from_the_payload_reaches_the_line(self):
        line, _ = mc_literal_report.describe({
            "boxesFound": True, "claimed": "<lora:secret:1> a woman on a bench",
            "config": "a secret", "autocompleteInstalled": True,
            "thirdPartyBoxes": "another secret", "extra": "and another"})

        assert "secret" not in line
        assert "bench" not in line

    def test_a_truthy_string_is_not_a_true(self):
        """`claimed: "no"` is a string, and a string is not an answer."""
        line, wrong = mc_literal_report.describe(report(claimed="no"))

        assert wrong is True
        assert "has not claimed" in line

    def test_anything_at_all_produces_a_line(self):
        for payload in (None, [], "", 0, {"unknown": "keys"}, {"claimed": None}):
            line, _ = mc_literal_report.describe(payload)

            assert line.startswith("Model Chain:")

    def test_it_never_raises(self):
        class Awkward(dict):
            def get(self, *args, **kwargs):
                raise RuntimeError("no")

        with pytest.raises(RuntimeError):
            Awkward().get("claimed")

        assert mc_literal_report.describe("not a dict")[0]


class TestTheLogLine:
    def test_a_fault_is_a_warning(self, caplog):
        with caplog.at_level(logging.DEBUG, logger="model_chain"):
            mc_literal_report.note(report(claimed=False))

        assert [r.levelno for r in caplog.records] == [logging.WARNING]

    def test_a_working_page_is_quiet(self, caplog):
        with caplog.at_level(logging.DEBUG, logger="model_chain"):
            mc_literal_report.note(report())

        assert [r.levelno for r in caplog.records] == [logging.DEBUG]


class TestTheRoute:
    class App:
        def __init__(self):
            self.routes = []

        def add_api_route(self, path, endpoint, methods=None):
            self.routes.append((path, endpoint, tuple(methods or ())))

    def test_it_registers_one_post(self):
        app = self.App()

        assert mc_literal_report.install(None, app) is True
        assert [(path, methods) for path, _, methods in app.routes] == [
            (mc_literal_report.ROUTE, ("POST",))]

    def test_registering_twice_leaves_one(self):
        app = self.App()
        mc_literal_report.install(None, app)

        class Named:
            path = mc_literal_report.ROUTE

        app.routes = [Named()]

        assert mc_literal_report.install(None, app) is True
        assert len(app.routes) == 1

    def test_a_build_without_an_app_is_not_a_crash(self):
        assert mc_literal_report.install(None, None) is False
        assert mc_literal_report.install(None, object()) is False

    def test_the_endpoint_logs_and_answers(self, caplog):
        app = self.App()
        mc_literal_report.install(None, app)
        endpoint = app.routes[0][1]

        with caplog.at_level(logging.DEBUG, logger="model_chain"):
            answer = endpoint(report(claimed=False))

        assert answer == {"logged": True}
        assert caplog.records

    def test_the_endpoint_swallows_a_bad_payload(self):
        app = self.App()
        mc_literal_report.install(None, app)
        endpoint = app.routes[0][1]

        assert endpoint("not a dict") == {"logged": True}
