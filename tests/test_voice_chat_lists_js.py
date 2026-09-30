"""Voice Chat's two voice lists, drawn from the server's own markup, in node.

The Voices panel's list and the character editor's picker are painted by
``javascript/voice_chat.js`` from ``/voice/voices``. Kokoro's official voices
say ``en-US`` or ``en-GB`` and were grouped by accent; PocketTTS's say only
``en``, and a list built from the two accents alone drew none of them: a Pocket
installation listed its custom voices and nothing else, in both places.

The markup is the server's (``mc_voice_ui``), built into the Voice Box
harness's element tree, so the check reads what a page with PocketTTS selected
paints -- not a copy of the grouping. These run under node, which is not a
Forge dependency, so they skip without it.
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
    return node;
}
TREES.forEach((markup) => markup.children.forEach((child) => {
    document.body.appendChild(build(child));
}));

globalThis.fetch = function (address) {
    const route = String(address).replace(/^.*model-chain\//, "");
    const answer = Object.prototype.hasOwnProperty.call(REPLIES, route)
        ? REPLIES[route] : {json: {ok: true}};
    const body = JSON.parse(JSON.stringify(answer.json));
    return Promise.resolve({ok: true, status: 200, headers: {get: () => null},
                            json: () => Promise.resolve(body),
                            text: () => Promise.resolve(JSON.stringify(body)),
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

function rows(selector, key) {
    const list = document.querySelector(selector);
    return list ? list.children.map((child) => child.hasAttribute(key)
        ? child.getAttribute(key) : "# " + child.textContent) : null;
}

await (async function () {
    advance(0);
    await flush(12);
    realConsole.log(JSON.stringify({
        listed: rows("[data-mc-voice-list]", "data-mc-voice-id"),
        picked: rows("[data-mc-voice-picker-list]", "data-mc-voice-pick"),
    }));
})();
"""


def voice(identifier: str, name: str, official: bool, language: str) -> dict:
    return {"id": identifier, "display_name": name, "label": name, "engine": "pocket",
            "official": official, "language": language, "editable": not official,
            "deletable": not official, "compatible": True, "has_source": False}


def painted(voices: list[dict]) -> dict:
    """Both lists as a page drawn for PocketTTS paints them from ``voices``."""
    engines.select("pocket")
    markup = [tree(mc_voice_ui.voices_html()),
              tree('<div id="mc-llm-chat-character-voice-list">'
                   + mc_voice_ui.character_voices_html() + "</div>")]
    answers = {"voice/voices": {"json": {"ok": True, "engine": "pocket",
                                         "default": "pocket:official:alba",
                                         "voices": voices}},
               "voice/profile": {"json": {"ok": True, "engine": "pocket"}}}
    harness = (HARNESS.replace("@@TREES@@", json.dumps(markup))
               .replace("@@ANSWERS@@", json.dumps(answers))
               .replace("@@SOURCE@@", SCRIPT.read_text(encoding="utf-8")))
    return _node(harness)


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
