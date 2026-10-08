"""TEMPORARY: the Generate box watcher's log route (mc_generate_watch).

A record from the browser becomes one line of this module's own shape: a
fixed set of keys, every value coerced and capped, printable ASCII only. A
move is a warning; the moments around it are ordinary lines.
"""

from __future__ import annotations

import logging

import pytest

import mc_generate_watch


def record(**extra):
    found = {"seq": 7, "t": 1700000000000, "kind": "dom.insertBefore",
             "box": "#txt2img_generate_box",
             "from": "#txt2img_gallery_container < #txt2img_results_panel",
             "to": "#txt2img_actions_column < #txt2img_toprow",
             "state": {"focus": True, "docked": False, "open": True, "full": True,
                       "hidden": False, "width": 1600, "height": 1000, "tab": "txt2img",
                       "active": "#txt2img_prompt",
                       "txt2img_generate_box": "#txt2img_generate_box < #txt2img_gallery_container"},
             "stack": ["insertBefore@http://host/file=javascript/x.js:10:3",
                       "m@http://host/assets/index-abc.js:1:2"]}
    found.update(extra)
    return found


class TestTheLine:
    def test_a_move_is_one_line_with_the_stack(self):
        line = mc_generate_watch.line_of(record())

        assert line.startswith(mc_generate_watch.PREFIX + " #7 t=1700000000000 dom.insertBefore")
        assert "from=#txt2img_gallery_container" in line
        assert "to=#txt2img_actions_column" in line
        assert "state{focus=true docked=false open=true full=true" in line
        assert "stack[insertBefore@http://host/file=javascript/x.js:10:3 | m@http://host/assets/index-abc.js:1:2]" in line
        assert "\n" not in line

    def test_nothing_unprintable_reaches_the_line(self):
        line = mc_generate_watch.line_of(record(kind="dom.\n\x00insert", box="#a\tbé"))

        assert "\n" not in line and "\x00" not in line and "\t" not in line
        assert "dom.??insert" in line
        assert "box=#a?b?" in line

    def test_every_field_is_capped(self):
        long = "x" * 5000
        line = mc_generate_watch.line_of(record(kind=long, box=long, stack=[long] * 100,
                                                state={"tab": long}))

        assert len(line) < (mc_generate_watch.LIMITS["kind"] + mc_generate_watch.LIMITS["field"]
                            + mc_generate_watch.LIMITS["state"]
                            + mc_generate_watch.STACK_LINES * (mc_generate_watch.LIMITS["stack_line"] + 3)
                            + 200)

    def test_a_value_of_the_wrong_type_is_named_not_printed(self):
        line = mc_generate_watch.line_of(record(box={"nested": "text"}, stack="not a list"))

        assert "box=dict" in line
        assert "stack[" not in line

    def test_anything_at_all_produces_a_line(self):
        for payload in (None, 3, "text", [], {"records": "no"}):
            assert mc_generate_watch.line_of(payload).startswith(mc_generate_watch.PREFIX)
        assert mc_generate_watch.lines_of(None) == []
        assert mc_generate_watch.lines_of({"records": "no"}) == []

    def test_a_post_is_capped_in_records(self):
        payload = {"records": [record()] * (mc_generate_watch.RECORDS + 50)}

        assert len(mc_generate_watch.lines_of(payload)) == mc_generate_watch.RECORDS


class TestTheLevels:
    def test_a_move_is_a_warning_and_a_moment_is_not(self, caplog):
        with caplog.at_level(logging.DEBUG, logger="model_chain"):
            written = mc_generate_watch.note({"records": [
                record(), record(kind="moved"), record(kind="replaced"),
                record(kind="focus.enter:before"), record(kind="event.resize")]})

        assert len(written) == 5
        assert [r.levelno for r in caplog.records] == [
            logging.WARNING, logging.WARNING, logging.WARNING, logging.INFO, logging.INFO]
        # The host's logger puts a time in front of every message.
        assert all(mc_generate_watch.PREFIX in r.getMessage() for r in caplog.records)

    def test_a_single_record_is_taken_as_well(self, caplog):
        with caplog.at_level(logging.DEBUG, logger="model_chain"):
            assert len(mc_generate_watch.note(record(kind="seen"))) == 1


class TestTheRoute:
    class App:
        def __init__(self):
            self.routes = []

        def add_api_route(self, path, endpoint, methods=None):
            self.routes.append((path, endpoint, tuple(methods or ())))

    def test_it_registers_one_post(self):
        app = self.App()

        assert mc_generate_watch.install(None, app) is True
        assert [(path, methods) for path, _, methods in app.routes] == [
            (mc_generate_watch.ROUTE, ("POST",))]

    def test_registering_twice_leaves_one(self):
        app = self.App()
        mc_generate_watch.install(None, app)

        class Named:
            path = mc_generate_watch.ROUTE

        app.routes = [Named()]

        assert mc_generate_watch.install(None, app) is True
        assert len(app.routes) == 1

    def test_the_endpoint_answers_with_the_count_and_never_raises(self, caplog):
        app = self.App()
        mc_generate_watch.install(None, app)
        endpoint = app.routes[0][1]
        with caplog.at_level(logging.DEBUG, logger="model_chain"):
            assert endpoint({"records": [record(), record(kind="seen")]}) == {"logged": 2}
            assert endpoint("nonsense") == {"logged": 0}

    def test_a_build_without_an_app_is_not_a_crash(self):
        assert mc_generate_watch.install(None, None) is False


class TestItIsRegistered:
    def test_the_script_registers_the_route_and_ships_the_watcher(self):
        from pathlib import Path

        root = Path(__file__).resolve().parent.parent
        source = (root / "scripts" / "model_chain.py").read_text(encoding="utf-8")
        assert "script_callbacks.on_app_started(mc_generate_watch.install)" in source
        assert (root / "javascript" / "model_chain_watch_generate.js").is_file()
        script = (root / "javascript" / "model_chain_watch_generate.js").read_text(encoding="utf-8")
        assert mc_generate_watch.ROUTE in script


@pytest.mark.parametrize("value, said", [
    (True, "true"), (False, "false"), (12, "12"), (None, ""), ("ok", "ok")])
def test_values_are_coerced(value, said):
    assert mc_generate_watch._text(value, 10) == said
