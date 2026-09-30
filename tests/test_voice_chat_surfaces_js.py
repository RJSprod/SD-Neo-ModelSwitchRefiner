"""Voice Chat's surfaces, drawn from the server's own markup, in node.

The markup is the server's (``mc_voice_ui``), built into the Voice Box
harness's element tree, and ``javascript/voice_chat.js`` runs over it against
answers shaped like the routes' -- so these read what a page with PocketTTS
selected paints, not a copy of the logic. They run under node, which is not a
Forge dependency, so they skip without it.

Two things a PocketTTS page got wrong:

* The Voices panel's list and the character editor's picker are painted from
  ``/voice/voices``. Kokoro's official voices say ``en-US`` or ``en-GB`` and
  were grouped by accent; PocketTTS's say only ``en``, and a list built from
  the two accents alone drew none of them: a Pocket installation listed its
  custom voices and nothing else, in both places.
* Changing a PocketTTS engine setting (precision, generation quality, the
  model) answers with Pocket's status under its own name beside the settings
  it applied, and the row painted the envelope: every line of it went blank,
  and nothing polled it back until the page was reloaded.
"""

from __future__ import annotations

import json
import shutil
from html.parser import HTMLParser
from pathlib import Path

import pytest

import mc_voice_engines as engines
import mc_voice_ui
from test_voice_box_js import DOM_JS, _node

SCRIPT = Path(__file__).resolve().parent.parent / "javascript" / "voice_chat.js"

pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")

VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param",
        "source", "track", "wbr"}


class _Tree(HTMLParser):
    """Markup as nested dicts the harness builds elements from."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.root = {"tag": "div", "attrs": {}, "children": []}
        self.stack = [self.root]

    def _node(self, tag, attrs):
        return {"tag": tag, "attrs": {name: ("" if value is None else value)
                                      for name, value in attrs}, "children": []}

    def handle_starttag(self, tag, attrs):
        node = self._node(tag, attrs)
        self.stack[-1]["children"].append(node)
        if tag not in VOID:
            self.stack.append(node)

    def handle_startendtag(self, tag, attrs):
        self.stack[-1]["children"].append(self._node(tag, attrs))

    def handle_endtag(self, tag):
        for index in range(len(self.stack) - 1, 0, -1):
            if self.stack[index]["tag"] == tag:
                del self.stack[index:]
                return

    def handle_data(self, data):
        if data:
            self.stack[-1]["children"].append({"text": data})


def tree(markup: str) -> dict:
    parser = _Tree()
    parser.feed(markup)
    parser.close()
    return parser.root


HARNESS = DOM_JS.replace("MICROPHONE", "true") + r"""
const REPLIES = @@ANSWERS@@;
const TREES = @@TREES@@;

function build(spec) {
    if (typeof spec.text === "string") return document.createTextNode(spec.text);
    const node = document.createElement(spec.tag);
    Object.keys(spec.attrs).forEach((name) => node.setAttribute(name, spec.attrs[name]));
    if (Object.prototype.hasOwnProperty.call(spec.attrs, "value")) node.value = spec.attrs.value;
    spec.children.forEach((child) => node.appendChild(build(child)));
    if (node.tagName === "SELECT") {
        const chosen = node.options.filter((option) => option.hasAttribute("selected"))[0]
            || node.options[0];
        node.value = chosen ? (chosen.getAttribute("value") || chosen.textContent) : "";
    }
    return node;
}
TREES.forEach((markup) => markup.children.forEach((child) => {
    document.body.appendChild(build(child));
}));

const requests = [];
globalThis.fetch = function (address, init) {
    const route = String(address).replace(/^.*model-chain\//, "");
    let body = null;
    try { body = JSON.parse((init && init.body) || "null"); } catch (error) { body = null; }
    requests.push({route, body});
    const answer = Object.prototype.hasOwnProperty.call(REPLIES, route)
        ? REPLIES[route] : {json: {ok: true}};
    const json = JSON.parse(JSON.stringify(answer.json));
    return Promise.resolve({ok: true, status: 200, headers: {get: () => null},
                            json: () => Promise.resolve(json),
                            text: () => Promise.resolve(JSON.stringify(json)),
                            arrayBuffer: () => Promise.resolve(new ArrayBuffer(64))});
};
globalThis.AudioContext = function () {
    return {state: "running", sampleRate: 48000, currentTime: 0, destination: {},
            resume() { return Promise.resolve(); }};
};
globalThis.addEventListener = () => {};
globalThis.removeEventListener = () => {};
const loaded = [];
globalThis.onUiLoaded = (fn) => loaded.push(fn);
globalThis.onAfterUiUpdate = () => {};

@@SOURCE@@

loaded.forEach((fn) => fn());

function q(selector) { return document.querySelector(selector); }
function text(selector) { const node = q(selector); return node ? node.textContent : null; }
function rows(selector, key) {
    const list = q(selector);
    return list ? list.children.map((child) => child.hasAttribute(key)
        ? child.getAttribute(key) : "# " + child.textContent) : null;
}
function report(extra) {
    realConsole.log(JSON.stringify(Object.assign({requests}, extra || {})));
}

await (async function () {
    advance(0);
    await flush(12);
@@SCENARIO@@
})();
"""


def run(markup: list[str], answers: dict, scenario: str) -> dict:
    harness = (HARNESS.replace("@@TREES@@", json.dumps([tree(one) for one in markup]))
               .replace("@@ANSWERS@@", json.dumps(answers))
               .replace("@@SOURCE@@", SCRIPT.read_text(encoding="utf-8"))
               .replace("@@SCENARIO@@", scenario))
    return _node(harness)


def voice(identifier: str, name: str, official: bool, language: str) -> dict:
    return {"id": identifier, "display_name": name, "label": name, "engine": "pocket",
            "official": official, "language": language, "editable": not official,
            "deletable": not official, "compatible": True, "has_source": False}


def painted(voices: list[dict]) -> dict:
    """Both lists as a page drawn for PocketTTS paints them from ``voices``."""
    engines.select("pocket")
    markup = [mc_voice_ui.voices_html(),
              '<div id="mc-llm-chat-character-voice-list">'
              + mc_voice_ui.character_voices_html() + "</div>"]
    answers = {"voice/voices": {"json": {"ok": True, "engine": "pocket",
                                         "default": "pocket:official:alba",
                                         "voices": voices}},
               "voice/profile": {"json": {"ok": True, "engine": "pocket"}}}
    return run(markup, answers, """
    report({listed: rows("[data-mc-voice-list]", "data-mc-voice-id"),
            picked: rows("[data-mc-voice-picker-list]", "data-mc-voice-pick")});
""")


class TestAnOfficialVoiceOfNeitherAccentIsListed:
    def test_pockets_official_voices_are_in_the_list_and_the_picker(self, voice_root):
        found = painted([voice("pocket:official:alba", "Alba", True, "en"),
                         voice("pocket:official:gb", "Brit", True, "en-GB"),
                         voice("pocket:clone:mine", "Mine", False, "")])

        assert found["listed"] == ["# Official — British English", "pocket:official:gb",
                                   "# Official", "pocket:official:alba",
                                   "# Custom", "pocket:clone:mine"]
        assert found["picked"] == ["", "# British", "pocket:official:gb",
                                   "# Official", "pocket:official:alba",
                                   "# Custom", "pocket:clone:mine"]

    def test_the_accents_keep_their_own_groups_and_nothing_is_listed_twice(self, voice_root):
        found = painted([voice("pocket:official:us", "Una", True, "en-US"),
                         voice("pocket:official:gb", "Brit", True, "en-GB")])

        assert found["listed"] == ["# Official — American English", "pocket:official:us",
                                   "# Official — British English", "pocket:official:gb"]
        assert found["picked"] == ["", "# American", "pocket:official:us",
                                   "# British", "pocket:official:gb"]


def pocket_status(**changes) -> dict:
    """``/voice/pocket``'s answer, the shape ``mc_voice_api.pocket_payload`` gives."""
    found = {"ok": True, "platform_supported": True, "runtime_ready": True,
             "speech_model_ready": True, "official_voices_ready": True,
             "cloning_ready": False, "runtime_message": "Installed",
             "model_message": "Installed — the English model",
             "cloning_message": "Not installed. Gated.", "message": "Installed.",
             "progress": {"pocket": {}}}
    found.update(changes)
    return found


class TestChangingAPocketSettingKeepsTheRowSaid:
    def test_the_row_is_painted_from_the_status_the_settings_route_returns(self, voice_root):
        engines.select("pocket")
        after = pocket_status(runtime_message="Installed — will restart at the next speech",
                              message="Precision changed.")
        answers = {"voice/pocket": {"json": pocket_status()},
                   "voice/engine/settings": {"json": {"ok": True, "engine": "pocket",
                                                      "settings": {"precision": "int8"},
                                                      "pocket": after}}}
        found = run([mc_voice_ui.settings_html()], answers, """
    const before = {runtime: text("[data-mc-voice-pocket-runtime]"),
                    status: text('[data-mc-voice-status="pocket"]')};
    const select = document.querySelector("[data-mc-voice-pocket-setting]");
    const name = select ? select.getAttribute("data-mc-voice-pocket-setting") : "";
    if (select) {
        select.value = select.options[select.options.length - 1].getAttribute("value")
            || select.options[select.options.length - 1].textContent;
        select.dispatchEvent({type: "change", bubbles: true, target: select,
                              preventDefault() {}});
    }
    advance(0);
    await flush(12);
    report({name, before, after: {runtime: text("[data-mc-voice-pocket-runtime]"),
                                  model: text("[data-mc-voice-pocket-model]"),
                                  cloning: text("[data-mc-voice-pocket-cloning]"),
                                  status: text('[data-mc-voice-status="pocket"]')}});
""")

        assert found["name"], "the Pocket row draws its engine settings"
        assert found["before"] == {"runtime": "Installed", "status": "Installed."}
        sent = [one for one in found["requests"] if one["route"] == "voice/engine/settings"]
        assert len(sent) == 1 and sent[0]["body"]["engine"] == "pocket"
        assert found["after"] == {"runtime": "Installed — will restart at the next speech",
                                  "model": "Installed — the English model",
                                  "cloning": "Not installed. Gated.",
                                  "status": "Precision changed."}
