"""The columns' swap, as far as node can hold it: the template that keeps each
column's width, the drag's clamp, what counts as a double tap, and the first
paint from what the browser kept.

Node has no layout; the swap itself, the drags and Forge's own scripts are
checked in Chromium (``tests/test_columns_swap.py``).
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "javascript" / "model_chain_columns.js"

HARNESS = """
const attributes = {};
const html = {
    getAttribute: (name) => (name in attributes ? attributes[name] : null),
    setAttribute(name, value) { attributes[name] = String(value); },
    removeAttribute(name) { delete attributes[name]; },
    hasAttribute(name) { return name in attributes; },
};
const listeners = [];
globalThis.document = {
    documentElement: html,
    body: {classList: {add() {}, remove() {}}},
    querySelector: () => null,
};
globalThis.window = globalThis;
globalThis.addEventListener = (type, handler, options) => listeners.push(
    type + (options === true || (options && options.capture) ? ":capture" : "") + (options && options.passive ? ":passive" : ""));
const store = __STORE__;
globalThis.localStorage = {
    getItem: (key) => (key in store ? store[key] : null),
    setItem: (key, value) => { store[key] = String(value); },
    removeItem: (key) => { delete store[key]; },
};
__OPTS__
const hooks = {loaded: [], changed: []};
globalThis.onUiLoaded = (fn) => hooks.loaded.push(fn);
globalThis.onOptionsChanged = (fn) => hooks.changed.push(fn);

SOURCE

const C = globalThis.mcColumns;
const out = {firstPaint: html.getAttribute("data-mc-columns-swapped"), listeners,
             hooks: {loaded: hooks.loaded.length, changed: hooks.changed.length}};
__SCENARIO__
out.store = store;
console.log(JSON.stringify(out));
"""


def run(scenario: str = "", store=None, opts=None) -> dict:
    if shutil.which("node") is None:
        pytest.skip("node is not installed")
    source = SCRIPT.read_text(encoding="utf-8")
    harness = (HARNESS.replace("__STORE__", json.dumps(store or {}))
               .replace("__OPTS__", "" if opts is None else "globalThis.opts = " + json.dumps(opts) + ";")
               .replace("__SCENARIO__", scenario)
               .replace("SOURCE", source))
    result = subprocess.run(["node", "--input-type=module"], input=harness, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout.strip().splitlines()[-1])


class TestTheTemplate:

    def test_forges_first_template_with_equal_halves_is_left_as_it_is(self):
        out = run("out.t = C.swappedTemplate('1fr 16px 1fr', '1fr 16px 1fr', [600, 16, 600]);")
        assert out["t"] is None

    def test_a_dragged_template_gives_the_first_track_the_width_of_the_column_coming_first(self):
        # The settings at 500 px (Forge's drag), the results 700 px: swapped,
        # the results come first and keep their 700.
        out = run("out.t = C.swappedTemplate('500px 16px 1fr', '1fr 16px 1fr', [500, 16, 699.6]);")
        assert out["t"] == "700px 16px 1fr"

    def test_forges_first_template_with_unequal_halves_is_written_in_pixels(self):
        out = run("out.t = C.swappedTemplate('1fr 16px 2fr', '1fr 16px 2fr', [400, 16, 800]);")
        assert out["t"] == "800px 16px 1fr"

    def test_nothing_laid_out_is_nothing_written(self):
        out = run("out.a = C.swappedTemplate('500px 16px 1fr', '1fr 16px 1fr', null);"
                  "out.b = C.swappedTemplate('500px 16px 1fr', '1fr 16px 1fr', [500, 16]);"
                  "out.c = C.swappedTemplate('500px 16px 1fr', '1fr 16px 1fr', [500, 16, 0]);")
        assert (out["a"], out["b"], out["c"]) == (None, None, None)


class TestTheDrag:

    def test_the_drag_follows_the_pointer_within_forges_minimums(self):
        out = run("out.right = C.dragWidth(600, 120, 1232, 320, 320);"
                  "out.left = C.dragWidth(600, -120.4, 1232, 320, 320);"
                  "out.floor = C.dragWidth(600, -1000, 1232, 320, 320);"
                  "out.ceiling = C.dragWidth(600, 1000, 1232, 320, 320);")
        assert out["right"] == 720 and out["left"] == 480
        assert out["floor"] == 320, "the column drawn first never narrower than its minimum"
        assert out["ceiling"] == 1232 - 320 - 16, "the column drawn last keeps its minimum and the handle"


class TestTheDoubleTap:

    def test_a_second_touch_within_the_double_tap_time_on_the_same_divider(self):
        out = run("const first = {tab: 'txt2img', time: 1000, moved: false};"
                  "out.soon = C.isDoubleTap(first, 'txt2img', 1000 + C.DOUBLE_TAP_MS);"
                  "out.late = C.isDoubleTap(first, 'txt2img', 1001 + C.DOUBLE_TAP_MS);"
                  "out.other = C.isDoubleTap(first, 'img2img', 1100);"
                  "out.dragged = C.isDoubleTap({tab: 'txt2img', time: 1000, moved: true}, 'txt2img', 1100);"
                  "out.none = C.isDoubleTap(null, 'txt2img', 1100);"
                  "out.ms = C.DOUBLE_TAP_MS;")
        assert out["soon"] is True and out["late"] is False
        assert out["other"] is False and out["dragged"] is False and out["none"] is False
        assert 200 <= out["ms"] <= 400, "about a platform's double-tap time, not a slow pair of taps"


class TestTheFirstPaint:

    def test_the_kept_choice_is_on_the_root_before_anything_is_built(self):
        out = run(store={"mc-columns-swapped:img2img": "1", "mc-columns-swapped:txt2img": "1"})
        assert out["firstPaint"] == "txt2img img2img"
        out = run(store={"mc-columns-swapped:img2img": "1"})
        assert out["firstPaint"] == "img2img"
        assert run()["firstPaint"] is None

    def test_the_setting_last_read_as_off_draws_nothing_until_forge_says(self):
        out = run(store={"mc-columns-swapped:txt2img": "1", "mc-columns-enabled": "0"})
        assert out["firstPaint"] is None
        # Forge's options, when they come, decide -- and are remembered.
        out = run("opts.model_chain_swap_columns = true; hooks.changed.forEach((fn) => fn());"
                  "out.after = document.documentElement.getAttribute('data-mc-columns-swapped');",
                  store={"mc-columns-swapped:txt2img": "1", "mc-columns-enabled": "0"}, opts={})
        assert out["after"] == "txt2img" and out["store"]["mc-columns-enabled"] == "1"
        out = run("opts.model_chain_swap_columns = false; hooks.changed.forEach((fn) => fn());"
                  "out.after = document.documentElement.getAttribute('data-mc-columns-swapped');",
                  store={"mc-columns-swapped:txt2img": "1"}, opts={})
        assert out["after"] is None and out["store"]["mc-columns-enabled"] == "0"
        assert out["store"]["mc-columns-swapped:txt2img"] == "1", "the choice is kept for when it is on again"

    def test_storage_that_throws_draws_forges_order(self):
        out = run("", store=None)
        assert out["firstPaint"] is None
        harness_store_throws = run(
            "globalThis.localStorage.getItem = () => { throw new Error('denied'); };"
            "C.render(); out.after = document.documentElement.getAttribute('data-mc-columns-swapped');",
            store={"mc-columns-swapped:txt2img": "1"})
        assert harness_store_throws["firstPaint"] == "txt2img"
        assert harness_store_throws["after"] is None


class TestTheListeners:

    def test_the_window_hears_presses_first_and_its_touch_listener_is_passive(self):
        out = run()
        assert "mousedown:capture" in out["listeners"] and "dblclick:capture" in out["listeners"]
        assert "touchstart:capture:passive" in out["listeners"], out["listeners"]
        assert out["hooks"] == {"loaded": 1, "changed": 1}

    def test_the_setting_and_the_store_keys(self):
        out = run("out.keys = [C.OPTION, C.STORE, C.ENABLED_STORE, C.ATTRIBUTE];")
        assert out["keys"] == ["model_chain_swap_columns", "mc-columns-swapped:", "mc-columns-enabled",
                               "data-mc-columns-swapped"]
