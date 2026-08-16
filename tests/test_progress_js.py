"""The styling script, executed rather than read.

The rest of the suite asserts things about this file as text. That was enough to
catch the constraints it has to honour and not enough to catch the one bug that
actually broke a WebUI: an exception escaping the ``requestProgress`` wrapper.

That failure is worth executing for because of where the host calls it from::

    const id = randomId();
    requestProgress(id, ...);          // the wrapper runs here
    const res = create_submit_args(arguments);
    res[0] = id;                       // the backend learns the id here
    return res;

``submit()`` is a Gradio ``_js`` handler and ``res[0]`` becomes ``id_task`` in
``wrap_gradio_gpu_call``. An exception here does not lose the decoration, it
aborts ``submit()`` before the task id is attached -- so the browser polls for
an id the backend never registered, ``/internal/progress`` answers
``active: false`` for the whole run, and the bar sits on "Waiting..." until the
generation ends and it vanishes. The images arrive normally and nothing in the
log connects the two.

These run under node, which is not a Forge dependency, so they skip without it.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "javascript" / "model_chain_progress.js"

pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")


HARNESS = """
// Minimal stand-ins for the globals the host provides.
const calls = [];
const classes = new Set();

globalThis.CSS = { supports: () => true };
globalThis.requestAnimationFrame = (fn) => fn();

function el(overrides = {}) {
    return Object.assign({
        classList: {
            toggle: (name, on) => (on ? classes.add(name) : classes.delete(name)),
            contains: (name) => classes.has(name),
            remove: (name) => classes.delete(name),
        },
        style: { setProperty() {}, removeProperty() {} },
        querySelectorAll: () => [],
        querySelector: () => null,
        appendChild() {},
        insertBefore() {},
        addEventListener() {},
        remove() {},
    }, overrides);
}

globalThis.document = {
    documentElement: el(),
    createElement: () => el(),
};
globalThis.setTimeout = () => 0;
globalThis.getComputedStyle = () => ({ content: "none" });

globalThis.opts = OPTS;

const callbacks = [];
globalThis.onOptionsAvailable = (fn) => callbacks.push(fn);
globalThis.onOptionsChanged = (fn) => {};
globalThis.onUiLoaded = (fn) => callbacks.push(fn);

// The host's own requestProgress, reduced to the fact that it was reached.
globalThis.requestProgress = function (id, container, gallery, atEnd, onProgress) {
    calls.push({ id: id });
    return "host-return-value";
};
globalThis.window = globalThis;

SOURCE

// Whatever the host would have run on load.
callbacks.forEach((fn) => fn());

const wrapped = globalThis.requestProgress.mcWrapped === true;

let threw = false;
let returned = null;
try {
    returned = globalThis.requestProgress("task(abc)", CONTAINER, null, null, null);
} catch (error) {
    threw = error.message;
}

console.log(JSON.stringify({ wrapped, threw, returned, reachedHost: calls.length, calls }));
"""


def run(options, container="el()"):
    source = SCRIPT.read_text()
    harness = (
        HARNESS.replace("OPTS", json.dumps(options))
        .replace("SOURCE", source)
        .replace("CONTAINER", container)
    )
    result = subprocess.run(
        ["node", "--input-type=module", "-e", harness],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout.strip().splitlines()[-1])


DEFAULTS = {
    "model_chain_style_enable": False,
    "model_chain_style_theme": "Flat",
    "model_chain_style_color": "",
    "model_chain_style_gradient": False,
    "model_chain_style_sheen": False,
    "model_chain_style_glow": False,
    "model_chain_style_complete": False,
}


class TestSubmitIsNeverBroken:
    """Nothing here may stop the host's call from happening or returning."""

    def test_the_hook_is_absent_until_something_needs_it(self):
        """The flash is off by default, so a default install has no hook.

        The cheapest possible guarantee: the code that could break submit() is
        not in the call path at all for anyone who has not opted in.
        """
        assert run(DEFAULTS)["wrapped"] is False

    def test_turning_the_flash_on_installs_it(self):
        options = {**DEFAULTS, "model_chain_style_enable": True, "model_chain_style_complete": True}

        assert run(options)["wrapped"] is True

    def test_the_host_is_reached_even_when_the_decoration_throws(self):
        """A container that raises on every DOM call, as a hostile theme might."""
        options = {**DEFAULTS, "model_chain_style_enable": True, "model_chain_style_complete": True}
        hostile = """{ get parentNode() { throw new Error("boom"); } }"""

        result = run(options, container=hostile)

        assert result["threw"] is False, "an exception escaped into submit()"
        assert result["reachedHost"] == 1, "the host's own call was skipped"

    def test_the_hosts_return_value_is_passed_back(self):
        options = {**DEFAULTS, "model_chain_style_enable": True, "model_chain_style_complete": True}

        assert run(options)["returned"] == "host-return-value"

    def test_the_task_id_reaches_the_host_unaltered(self):
        """It is the only argument whose value the backend depends on."""
        options = {**DEFAULTS, "model_chain_style_enable": True, "model_chain_style_complete": True}

        assert run(options)["calls"][0]["id"] == "task(abc)"

    def test_styling_alone_does_not_install_the_hook(self):
        """Colours and themes are CSS; only the flash needs to see the lifecycle."""
        options = {**DEFAULTS, "model_chain_style_enable": True, "model_chain_style_theme": "Neon"}

        assert run(options)["wrapped"] is False
