"""VibeVoice's Voice Chat surfaces, executed rather than read.

The Settings row, the clone form, the voice list and the delivery sliders are
drawn by the server exactly as a page receives them -- ``mc_voice_ui``'s
``settings_html`` and ``voices_html`` with VibeVoice selected, against the fakes
of ``tests/test_voice_vibevoice_speech.py`` -- built into the element tree of
``tests/test_voice_box_js.py``, and wired by the real ``javascript/voice_chat.js``.
What the routes answer is what ``mc_voice_api`` answers for the same fakes. So a
selector the script looks for and the markup does not carry, or a field the
script reads and the payload does not have, fails here rather than on somebody's
settings page.

What is VibeVoice's, and so what is checked:

    the panel      asks its own route only while it is on screen and the page is
                   not hidden, one request at a time, each with a deadline, and
                   paints the installer's parts, the cards with their roles, the
                   precisions and the LoRA library from the answer;
    a setting      goes through the engine-neutral settings route, naming the
                   engine the surface was drawn for;
    cloning        Create follows the installation (the 7B, not the Realtime
                   model); the preview is one request with a deadline long enough
                   for a card turn and a cold 7B, which a second press waits for
                   rather than repeats; Save makes a Voice Box sample the voice
                   list shows; Discard forgets it at once;
    the list       is grouped by the model a voice belongs to;
    delivery       a slider that follows the model's own value is sent as that,
                   and Reset puts it back;
    a silence      on this engine is a reply waiting for its card, and says so.

These run under node, which is not a Forge dependency, so they skip without it.
"""

from __future__ import annotations

import json
import shutil
from html.parser import HTMLParser
from pathlib import Path

import pytest

import mc_voice_api as api
import mc_voice_box as box
import mc_voice_engines as engines
import mc_voice_ui
import mc_voice_vibevoice_chat as chat
from test_voice_box_js import DOM_JS, _node
from test_voice_chat_js import run as run_composer
from test_voice_chat_js import status_answer
from test_voice_vibevoice_speech import MODEL_7B, tone_wav, vibevoice  # noqa: F401

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "javascript" / "voice_chat.js"

pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")

PLACEHOLDERS = ("@@SOURCE@@", "@@SCENARIO@@", "@@TREES@@", "@@ANSWERS@@", "@@HELD@@",
                "@@DECODE_SECONDS@@")

VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param",
        "source", "track", "wbr"}


class _Tree(HTMLParser):
    """Markup as nested dicts the harness builds elements from. Nothing clever:
    the server's markup is well formed, and a void element never has children."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.root = {"tag": "div", "attrs": {}, "children": []}
        self.stack = [self.root]

    def _node(self, tag, attrs):
        return {"tag": tag, "attrs": {name: ("" if value is None else value)
                                      for name, value in attrs},
                "children": []}

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
// --- the harness for VibeVoice's surfaces --------------------------------- //

const REPLIES = @@ANSWERS@@;
const HELD = new Set(@@HELD@@);
const TREES = @@TREES@@;
const DECODE_SECONDS = @@DECODE_SECONDS@@;

// The server's markup, built into the element tree attribute for attribute.
function build(spec) {
    if (typeof spec.text === "string") return document.createTextNode(spec.text);
    const node = document.createElement(spec.tag);
    const has = (name) => Object.prototype.hasOwnProperty.call(spec.attrs, name);
    Object.keys(spec.attrs).forEach((name) => node.setAttribute(name, spec.attrs[name]));
    if (has("value")) node.value = spec.attrs.value;
    if (has("disabled")) node.disabled = true;
    if (has("checked")) node.checked = true;
    spec.children.forEach((child) => node.appendChild(build(child)));
    if (node.tagName === "OPTION" && !has("value")) node.value = node.textContent;
    if (node.tagName === "SELECT") {
        const chosen = node.options.filter((option) => option.hasAttribute("selected"))[0]
            || node.options[0];
        node.value = chosen ? chosen.value : "";
    }
    return node;
}
TREES.forEach((markup) => markup.children.forEach((child) => {
    document.body.appendChild(build(child));
}));

// Every request: its route, its body, and whether it came with a deadline.
const requests = [];
const held = [];
function formEntries(body) {
    return Array.from(body.entries()).map(function (pair) {
        const value = pair[1];
        return [pair[0], typeof value === "string" ? value
                : {type: value.type || "", size: value.size || 0, name: value.name || ""}];
    });
}
globalThis.fetch = function (address, init) {
    const options = init || {};
    const route = String(address).replace(/^.*model-chain\//, "");
    const record = {route, method: options.method || "GET",
                    key: (options.headers || {})["X-Model-Chain-Voice"] || "",
                    signal: !!options.signal, aborted: false, json: null, form: null};
    if (typeof options.body === "string") {
        try { record.json = JSON.parse(options.body); } catch (error) { record.json = options.body; }
    } else if (options.body && typeof options.body.entries === "function") {
        record.form = formEntries(options.body);
    }
    requests.push(record);
    const answer = Object.prototype.hasOwnProperty.call(REPLIES, route)
        ? REPLIES[route] : {json: {ok: true}};
    const respond = function () {
        const status = answer.status || 200;
        const body = JSON.parse(JSON.stringify(answer.json === undefined ? {} : answer.json));
        return {ok: status < 400, status, headers: {get: () => null},
                json: () => Promise.resolve(body),
                text: () => Promise.resolve(JSON.stringify(body)),
                arrayBuffer: () => Promise.resolve(new ArrayBuffer(64))};
    };
    if (!HELD.has(route)) return Promise.resolve(respond());
    return new Promise(function (resolve, reject) {
        held.push({route, record, release: () => resolve(respond())});
        if (options.signal) {
            options.signal.addEventListener("abort", function () {
                record.aborted = true;
                const error = new Error("The operation was aborted.");
                error.name = "AbortError";
                reject(error);
            });
        }
    });
};
// Answer every held request to `route`, and let the next ones through.
function release(route) {
    HELD.delete(route);
    held.filter((entry) => entry.route === route).forEach((entry) => entry.release());
}

// Web Audio: a decoder that says every file is DECODE_SECONDS long, and sources
// that record having been started.
const played = [];
globalThis.AudioContext = function () {
    return {
        state: "running",
        sampleRate: 48000,
        currentTime: 0,
        destination: {},
        resume() { return Promise.resolve(); },
        createMediaStreamSource() { return {connect() {}, disconnect() {}}; },
        createScriptProcessor() { return {connect() {}, disconnect() {}, onaudioprocess: null}; },
        createGain() { return {gain: {value: 1}, connect() {}, disconnect() {}}; },
        createBuffer(channels, length, rate) {
            const data = new Float32Array(length);
            return {numberOfChannels: channels, length, sampleRate: rate, duration: length / rate,
                    getChannelData: () => data};
        },
        createBufferSource() {
            const source = {buffer: null, onended: null, connect() {}, disconnect() {},
                            start() { played.push(source); }, stop() {}};
            return source;
        },
        decodeAudioData(buffer, ok) {
            const rate = 24000;
            const length = Math.round(DECODE_SECONDS * rate);
            const data = new Float32Array(length);
            for (let index = 0; index < length; index += 1) data[index] = 0.3 * Math.sin(index / 7);
            const decoded = {numberOfChannels: 1, length, sampleRate: rate,
                             duration: length / rate, getChannelData: () => data};
            if (ok) ok(decoded);
            return Promise.resolve(decoded);
        },
    };
};

// A promise nobody caught is a failure of the page, not of the harness: kept,
// and reported, rather than allowed to end the process.
const rejections = [];
process.on("unhandledRejection", (reason) => {
    rejections.push(String((reason && reason.stack) || reason));
});

// The window's own events (page hide, focus) are the composer's business.
globalThis.addEventListener = () => {};
globalThis.removeEventListener = () => {};

const loaded = [];
globalThis.onUiLoaded = (fn) => loaded.push(fn);
globalThis.onAfterUiUpdate = () => {};

@@SOURCE@@

loaded.forEach((fn) => fn());

function q(selector) { return document.querySelector(selector); }
function row() { return q('[data-mc-voice-kind="vibevoice"]'); }
function sent(route) { return requests.filter((request) => request.route === route); }
function fire(node, type) {
    node.dispatchEvent({type, bubbles: true, target: node, preventDefault() {}});
}
function change(node, value) { node.value = value; fire(node, "change"); }
async function step(ms) {
    advance(ms || 0);
    await flush(8);
}
// A recording chosen from a file: the decoder says how long it is.
async function chooseRecording(name) {
    const form = q("[data-mc-voice-vibevoice-form]");
    q("[data-mc-voice-vibevoice-name]").value = name === undefined ? "Ada" : name;
    const file = q("[data-mc-voice-vibevoice-file]");
    file.files = [new File([new Uint8Array(64)], "ada.wav", {type: "audio/wav"})];
    fire(file, "change");
    await flush(8);
    return form;
}
function text(selector) {
    const node = q(selector);
    return node ? node.textContent : null;
}
function options(selector) {
    const node = q(selector);
    return node ? node.options.map((option) => [option.value, option.textContent]) : null;
}
function listed() {
    const list = q("[data-mc-voice-list]");
    return list.children.map((child) => child.getAttribute("data-mc-voice-id")
                             || ("# " + child.textContent));
}
function report(extra) {
    realConsole.log(JSON.stringify(Object.assign({
        requests, warnings, rejections, played: played.length, pending: pendingDelays(),
    }, extra || {})));
}

await (async function () {
@@SCENARIO@@
})();
"""


class Page:
    """The drawn surfaces and what the routes answer, for one scenario."""

    def __init__(self, fakes):
        self.fakes = fakes

    def trees(self) -> list:
        return [tree(mc_voice_ui.settings_html()), tree(mc_voice_ui.voices_html())]

    def settings_only(self) -> list:
        """The Settings row without the voice list, whose own answer would
        otherwise say some of what the panel's must."""
        return [tree(mc_voice_ui.settings_html())]

    def answers(self) -> dict:
        return {
            "voice/vibevoice": {"json": api.vibevoice_payload()},
            "voice/voices": {"json": api.voices_payload()},
            "voice/profile": {"json": api.profile_payload(None, "vibevoice")},
            "voice/status": {"json": api.status_payload()},
        }

    def run(self, scenario: str, *, trees=None, answers=None, held=(),
            decode_seconds: float = 10.0) -> dict:
        drawn = self.trees() if trees is None else trees
        table = self.answers()
        table.update(answers or {})
        harness = (HARNESS
                   .replace("@@TREES@@", json.dumps(drawn))
                   .replace("@@ANSWERS@@", json.dumps(table))
                   .replace("@@HELD@@", json.dumps(list(held)))
                   .replace("@@DECODE_SECONDS@@", json.dumps(decode_seconds))
                   .replace("@@SOURCE@@", SCRIPT.read_text(encoding="utf-8"))
                   .replace("@@SCENARIO@@", scenario))
        found = _node(harness)
        # Nothing of VibeVoice's may throw, whatever the unrelated rows do with
        # the harness's plain answers.
        for line in found["warnings"] + found["rejections"]:
            assert "vibevoice" not in line.lower(), line
        return found


@pytest.fixture
def page(host, vibevoice):
    engines.select("vibevoice")
    yield Page(vibevoice)
    chat.discard_preview()


def routes(found: dict) -> list:
    return [request["route"] for request in found["requests"]]


def of(found: dict, route: str) -> list:
    return [request for request in found["requests"] if request["route"] == route]


def test_the_files_under_test_avoid_the_placeholders():
    """The harness substitutes these; a script containing one would be rewritten
    mid-token into a syntax error far from anything."""
    source = SCRIPT.read_text(encoding="utf-8")
    for word in PLACEHOLDERS:
        assert word not in source, word


# --------------------------------------------------------------------------- #
# The panel's own poll
# --------------------------------------------------------------------------- #


class TestThePanelPoll:
    def test_it_asks_its_own_route_once_with_the_page_token_and_a_deadline(self, page):
        found = page.run("""
            await step(0);
            report({status: text('[data-mc-voice-status="vibevoice"]')});
        """)

        asked = of(found, "voice/vibevoice")
        assert len(asked) == 1
        assert asked[0]["method"] == "POST"
        assert asked[0]["signal"] is True
        assert asked[0]["key"] == api.session_token()
        assert found["status"] == chat.status().message

    def test_it_looks_again_every_twenty_seconds_when_nothing_is_happening(self, page):
        """Installs happen in another tab, so asking is the only way to learn of
        one -- and a request a second and a half for as long as the page is open
        is what section 33 forbids."""
        found = page.run("""
            await step(0);
            await step(19999);
            const early = sent("voice/vibevoice").length;
            await step(1);
            report({early});
        """)

        assert found["early"] == 1
        assert len(of(found, "voice/vibevoice")) == 2

    def test_while_the_voice_box_installs_it_says_so_and_looks_more_often(self, page):
        installing = dict(api.vibevoice_payload(), progress={
            "running": True, "text": "Downloading VibeVoice 7B — 41%", "fraction": 0.41,
            "failed": False, "model": MODEL_7B})
        found = page.run("""
            await step(0);
            const said = text('[data-mc-voice-status="vibevoice"]');
            await step(1500);
            report({said});
        """, answers={"voice/vibevoice": {"json": installing}})

        assert found["said"] == "Downloading VibeVoice 7B — 41%"
        assert len(of(found, "voice/vibevoice")) == 2

    def test_it_paints_the_installers_parts_and_models(self, page):
        found = page.run("""
            await step(0);
            report({runtime: text('[data-mc-voice-vibevoice-part="runtime"]'),
                    big: text('[data-mc-voice-vibevoice-model="vibevoice-7b"]'),
                    small: text('[data-mc-voice-vibevoice-model="vibevoice-realtime-0.5b"]')});
        """, answers={"voice/vibevoice": {"json": dict(
            api.vibevoice_payload(), runtime_message="Installed — painted by the poll.")}})

        assert found["runtime"] == "Installed — painted by the poll."
        assert found["big"] == "VibeVoice 7B — Installed."
        assert found["small"] == "VibeVoice Realtime 0.5B — Installed."

    def test_it_offers_the_cards_with_their_roles_and_the_precisions_and_loras(self, page):
        found = page.run("""
            await step(0);
            report({cards: options('[data-mc-voice-vibevoice-setting="card_uuid"]'),
                    card: q('[data-mc-voice-vibevoice-setting="card_uuid"]').value,
                    precisions: options('[data-mc-voice-vibevoice-setting="precision"]'),
                    loras: options('[data-mc-voice-vibevoice-setting="lora_id"]'),
                    strength: text("[data-mc-voice-vibevoice-strength]")});
        """)

        assert found["cards"] == [
            ["", "The Voice Box's card"],
            ["GPU-11111111-2222-3333-4444-555555555555",
             "NVIDIA GeForce RTX 3090 — the image model's card"],
            ["GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee", "NVIDIA GeForce RTX 5090 — WanGP's card"],
        ]
        assert found["card"] == "GPU-11111111-2222-3333-4444-555555555555"
        assert [value for value, _label in found["precisions"]] == ["bf16", "int8", "nf4"]
        assert found["precisions"][1][1] == "8-bit — about 12.0 GB on the card"
        assert found["loras"] == [["", "None"], ["warm01", "Warm"]]
        assert found["strength"] == "1.00"

    def test_a_row_nobody_can_see_is_asked_nothing_until_it_can_be_seen(self, page):
        found = page.run("""
            const panel = row();
            panel.offsetParent = null;
            panel.clientWidth = 0;
            panel.clientHeight = 0;
            await step(0);
            await step(5000);
            const whileHidden = sent("voice/vibevoice").length;
            panel.offsetParent = {};
            panel.clientWidth = 240;
            await step(1000);
            report({whileHidden});
        """)

        assert found["whileHidden"] == 0
        assert len(of(found, "voice/vibevoice")) == 1

    def test_a_hidden_page_is_asked_nothing_and_looked_at_again(self, page):
        found = page.run("""
            await step(0);
            await step(20000);
            const before = sent("voice/vibevoice").length;
            document.hidden = true;
            await step(20000);
            await step(5000);
            const whileHidden = sent("voice/vibevoice").length;
            document.hidden = false;
            await step(1000);
            report({before, whileHidden});
        """)

        assert found["before"] == 2
        assert found["whileHidden"] == 2
        assert len(of(found, "voice/vibevoice")) == 3

    def test_a_request_that_never_answers_is_given_up_at_its_deadline(self, page):
        """A request without a deadline is a bug on this page: the stalled HTTP/2
        connection held eleven of them for ever. Aborted at fifteen seconds, and
        the panel asks again on its ordinary schedule."""
        found = page.run("""
            await step(0);
            await step(14999);
            const before = sent("voice/vibevoice").map((r) => r.aborted);
            await step(1);
            const after = sent("voice/vibevoice").map((r) => r.aborted);
            release("voice/vibevoice");
            await step(20000);
            report({before, after,
                    said: text('[data-mc-voice-status="vibevoice"]')});
        """, held=["voice/vibevoice"])

        assert found["before"] == [False]
        assert found["after"] == [True]
        assert len(of(found, "voice/vibevoice")) == 2
        # The answer that finally came is painted; the late one never was.
        assert found["said"] == chat.status().message

    def test_never_a_second_request_beside_one_in_flight(self, page):
        """A Save, a refused setting and the schedule all ask for a look; while one
        is on its way, none of them adds another."""
        found = page.run("""
            await step(0);
            const precision = q('[data-mc-voice-vibevoice-setting="precision"]');
            change(precision, "int8");
            await step(0);
            await step(0);
            report({asked: sent("voice/vibevoice").length});
        """, held=["voice/vibevoice"],
            answers={"voice/engine/settings": {"status": 400, "json": {
                "ok": False, "error": "VibeVoice 7B cannot run at that precision."}}})

        assert found["asked"] == 1


# --------------------------------------------------------------------------- #
# Engine settings
# --------------------------------------------------------------------------- #


class TestASetting:
    def test_it_is_posted_to_the_engine_neutral_route_naming_the_engine(self, page):
        drawn = page.trees()
        changed = api.engine_settings({"precision": "int8"}, engine="vibevoice")
        found = page.run("""
            await step(0);
            const precision = q('[data-mc-voice-vibevoice-setting="precision"]');
            change(precision, "int8");
            await step(0);
            report({precision: precision.value});
        """, trees=drawn, answers={"voice/engine/settings": {"json": changed}})

        posted = of(found, "voice/engine/settings")
        assert len(posted) == 1
        assert posted[0]["json"] == {"engine": "vibevoice", "values": {"precision": "int8"}}
        assert posted[0]["signal"] is True
        assert found["precision"] == "int8"

    def test_the_strength_follows_the_slider_and_is_sent_as_a_number_when_let_go(self, page):
        found = page.run("""
            await step(0);
            const strength = q('[data-mc-voice-vibevoice-setting="lora_scale"]');
            strength.value = "1.4";
            fire(strength, "input");
            const moving = {shown: text("[data-mc-voice-vibevoice-strength]"),
                            posted: sent("voice/engine/settings").length};
            fire(strength, "change");
            await step(0);
            report({moving});
        """)

        assert found["moving"] == {"shown": "1.40", "posted": 0}
        posted = of(found, "voice/engine/settings")
        assert posted[0]["json"] == {"engine": "vibevoice", "values": {"lora_scale": 1.4}}

    def test_a_refused_setting_says_why_and_looks_again(self, page):
        found = page.run("""
            await step(0);
            change(q('[data-mc-voice-vibevoice-setting="lora_id"]'), "gone");
            await step(0);
            const said = text('[data-mc-voice-status="vibevoice"]');
            await step(0);
            report({said});
        """, answers={"voice/engine/settings": {"status": 400, "json": {
            "ok": False, "error": "That LoRA is no longer in the library."}}})

        assert found["said"] == "That LoRA is no longer in the library."
        # What the server holds now, rather than what was chosen.
        assert len(of(found, "voice/vibevoice")) == 2


# --------------------------------------------------------------------------- #
# Making a voice from a recording
# --------------------------------------------------------------------------- #


class TestCreateFollowsTheInstallation:
    def test_without_the_7b_it_is_disabled_and_says_so_until_a_poll_says_otherwise(
            self, page, vibevoice):
        vibevoice.installer.installed[MODEL_7B] = False
        drawn = page.trees()
        waiting = page.answers()
        vibevoice.installer.installed[MODEL_7B] = True
        found = page.run("""
            const create = q("[data-mc-voice-vibevoice-create]");
            const first = {disabled: create.disabled,
                           said: text("[data-mc-voice-vibevoice-clone-status]")};
            await step(0);
            const polled = {disabled: create.disabled,
                            said: text("[data-mc-voice-vibevoice-clone-status]")};
            REPLIES["voice/vibevoice"] = LATER;
            await step(20000);
            report({first, polled, disabled: create.disabled,
                    said: text("[data-mc-voice-vibevoice-clone-status]")});
        """.replace("LATER", json.dumps({"json": api.vibevoice_payload()})),
            trees=drawn, answers=waiting)

        assert found["first"]["disabled"] is True
        assert "needs the VibeVoice 7B model" in found["first"]["said"]
        assert "cannot make one" in found["first"]["said"]
        assert found["polled"] == found["first"]
        # Installed in the Voice Box tab meanwhile: enabled without a reload, and
        # the sentence the poll wrote is taken back.
        assert found["disabled"] is False
        assert found["said"] == ""

    def test_the_window_is_the_engines_own_from_its_payload(self, page):
        """The fallback window is Sopro's five to twenty; VibeVoice's is the Voice
        Box's, and it arrives with the panel's answer."""
        hints = chat.clone_hints()
        found = page.run("""
            await step(0);
            await chooseRecording("Ada");
            q("[data-mc-voice-vibevoice-create]").click();
            await step(0);
            report({said: text("[data-mc-voice-vibevoice-clone-status]")});
        """, trees=page.settings_only(), decode_seconds=float(hints["min_seconds"]) - 1.0)

        low, high = int(hints["min_seconds"]), int(hints["max_seconds"])
        assert found["said"].startswith(f"VibeVoice clones from {low} to {high} seconds.")
        assert not of(found, "voice/clone/preview")


class TestThePreview:
    def test_it_is_one_request_that_carries_the_name_the_engine_and_the_selection(self, page):
        found = page.run("""
            await step(0);
            await chooseRecording("Ada");
            const create = q("[data-mc-voice-vibevoice-create]");
            create.click();
            await step(0);
            report({disabled: create.disabled,
                    said: text("[data-mc-voice-vibevoice-clone-status]")});
        """, held=["voice/clone/preview"])

        posted = of(found, "voice/clone/preview")
        assert len(posted) == 1
        assert posted[0]["signal"] is True
        form = dict((name, value) for name, value in posted[0]["form"])
        assert form["name"] == "Ada"
        assert form["engine"] == "vibevoice"
        assert form["reference"]["type"] == "audio/wav"
        assert form["reference"]["size"] > 44
        assert found["disabled"] is True
        assert found["said"].startswith("Preparing the voice… VibeVoice asks for its turn on "
                                        "the graphics card")

    def test_a_second_press_waits_for_the_first_rather_than_asking_again(self, page):
        """Every preview is a card turn and maybe a cold 7B: a second request would
        be a second turn in the queue behind the first."""
        found = page.run("""
            await step(0);
            await chooseRecording("Ada");
            const create = q("[data-mc-voice-vibevoice-create]");
            create.click();
            await step(0);
            create.disabled = false;
            create.click();
            await step(0);
            report({});
        """, held=["voice/clone/preview"])

        assert len(of(found, "voice/clone/preview")) == 1

    def test_its_deadline_is_six_minutes_and_then_it_says_so(self, page):
        found = page.run("""
            await step(0);
            await chooseRecording("Ada");
            const create = q("[data-mc-voice-vibevoice-create]");
            create.click();
            await step(0);
            await step(6 * 60 * 1000 - 1);
            const before = {aborted: sent("voice/clone/preview")[0].aborted,
                            disabled: create.disabled};
            await step(1);
            report({before, aborted: sent("voice/clone/preview")[0].aborted,
                    disabled: create.disabled,
                    said: text("[data-mc-voice-vibevoice-clone-status]")});
        """, held=["voice/clone/preview"])

        assert found["before"] == {"aborted": False, "disabled": True}
        assert found["aborted"] is True
        assert found["disabled"] is False
        assert found["said"].startswith("VibeVoice did not answer within six minutes")

    def test_while_it_is_being_made_a_poll_does_not_hand_the_button_back(self, page):
        found = page.run("""
            await step(0);
            await chooseRecording("Ada");
            const create = q("[data-mc-voice-vibevoice-create]");
            create.click();
            await step(0);
            await step(20000);
            report({polls: sent("voice/vibevoice").length, disabled: create.disabled});
        """, held=["voice/clone/preview"])

        assert found["polls"] == 2
        assert found["disabled"] is True

    def test_a_ready_preview_is_played_and_offered_for_save(self, page):
        made = api.clone_preview("Ada", tone_wav(5.0), engine="vibevoice")
        found = page.run("""
            await step(0);
            await chooseRecording("Ada");
            q("[data-mc-voice-vibevoice-create]").click();
            await step(0);
            const preview = q("[data-mc-voice-vibevoice-preview]");
            report({hidden: preview.hidden,
                    note: text("[data-mc-voice-vibevoice-preview-note]"),
                    said: text("[data-mc-voice-vibevoice-clone-status]")});
        """, answers={"voice/clone/preview": {"json": made}})

        assert found["hidden"] is False
        assert found["note"] == "“Ada” is ready to listen to. It is not saved yet."
        assert found["said"] == "Ready. Listen, then Save voice or Discard."
        assert found["played"] == 1

    def test_save_sends_the_token_and_the_new_sample_is_in_the_list(self, page):
        # What the page was told before the voice existed, so the list can
        # only learn of it from Save's own answer.
        drawn = page.trees()
        before = page.answers()
        made = api.clone_preview("Ada", tone_wav(5.0), engine="vibevoice")
        kept = api.clone_save(made["token"], engine="vibevoice")
        found = page.run("""
            await step(0);
            await chooseRecording("Ada");
            q("[data-mc-voice-vibevoice-create]").click();
            await step(0);
            const polls = sent("voice/vibevoice").length;
            q("[data-mc-voice-vibevoice-preview-save]").click();
            await step(0);
            await step(0);
            report({polls, hidden: q("[data-mc-voice-vibevoice-preview]").hidden,
                    said: text("[data-mc-voice-vibevoice-clone-status]"),
                    listed: listed(), after: sent("voice/vibevoice").length});
        """, trees=drawn, answers=dict(before, **{"voice/clone/preview": {"json": made},
                                                  "voice/clone/save": {"json": kept}}))

        saved = of(found, "voice/clone/save")
        assert len(saved) == 1
        assert saved[0]["json"] == {"token": made["token"], "engine": "vibevoice"}
        assert saved[0]["signal"] is True
        assert found["hidden"] is True
        assert found["said"] == "Saved Ada. It is a sample in the Voice Box library too."
        assert kept["voice"]["id"] in found["listed"]
        # And the panel looks again, so the sample count and Create follow.
        assert found["after"] == found["polls"] + 1

    def test_a_refused_save_keeps_the_preview(self, page):
        made = api.clone_preview("Ada", tone_wav(5.0), engine="vibevoice")
        found = page.run("""
            await step(0);
            await chooseRecording("Ada");
            q("[data-mc-voice-vibevoice-create]").click();
            await step(0);
            q("[data-mc-voice-vibevoice-preview-save]").click();
            await step(0);
            report({hidden: q("[data-mc-voice-vibevoice-preview]").hidden,
                    said: text("[data-mc-voice-vibevoice-clone-status]")});
        """, answers={"voice/clone/preview": {"json": made},
                      "voice/clone/save": {"status": 409, "json": {
                          "ok": False, "error": "That preview is no longer pending."}}})

        assert found["hidden"] is False
        assert found["said"] == "That preview is no longer pending."

    def test_discard_forgets_it_at_once_and_tells_the_server(self, page):
        made = api.clone_preview("Ada", tone_wav(5.0), engine="vibevoice")
        found = page.run("""
            await step(0);
            await chooseRecording("Ada");
            q("[data-mc-voice-vibevoice-create]").click();
            await step(0);
            q("[data-mc-voice-vibevoice-preview-discard]").click();
            const at_once = {hidden: q("[data-mc-voice-vibevoice-preview]").hidden,
                             said: text("[data-mc-voice-vibevoice-clone-status]")};
            await step(0);
            report({at_once});
        """, answers={"voice/clone/preview": {"json": made}}, held=["voice/clone/discard"])

        assert found["at_once"] == {"hidden": True, "said": "Discarded."}
        discarded = of(found, "voice/clone/discard")
        assert discarded[0]["json"] == {"token": made["token"], "engine": "vibevoice"}
        assert discarded[0]["signal"] is True


# --------------------------------------------------------------------------- #
# The voice list and the delivery sliders
# --------------------------------------------------------------------------- #


class TestTheVoiceList:
    def test_it_is_grouped_by_the_model_a_voice_belongs_to(self, page):
        made = box.add_sample(tone_wav(4.0), title="Ada")
        found = page.run("""
            await step(0);
            report({listed: listed(), current: text("[data-mc-voice-current]")});
        """)

        assert found["listed"] == [
            "# Preset voices — English",
            "vibevoice:preset:en-Carter_man",
            "vibevoice:preset:en-Emma_woman",
            "# Preset voices — other languages (experimental)",
            "vibevoice:preset:de-Spk0_man",
            "# Voice Box samples",
            f"vibevoice:sample:{made['id']}",
        ]
        assert found["current"] == "Default voice: Carter (English, man)"

    def test_the_character_picker_is_grouped_the_same_way(self, page):
        """The compact list in the character screen: the default first, then the
        same three groups under shorter headings."""
        made = box.add_sample(tone_wav(4.0), title="Ada")
        drawn = page.trees() + [tree('<div id="mc-llm-chat-character-voice-list">'
                                     + mc_voice_ui.character_voices_html() + '</div>')]
        found = page.run("""
            await step(0);
            const list = q("[data-mc-voice-picker-list]");
            report({picked: list.children.map((child) =>
                child.hasAttribute("data-mc-voice-pick")
                    ? child.getAttribute("data-mc-voice-pick")
                    : "# " + child.textContent)});
        """, trees=drawn)

        assert found["picked"] == [
            "",
            "# English",
            "vibevoice:preset:en-Carter_man",
            "vibevoice:preset:en-Emma_woman",
            "# Other languages",
            "vibevoice:preset:de-Spk0_man",
            "# Samples",
            f"vibevoice:sample:{made['id']}",
        ]


    def test_an_official_voice_of_neither_accent_is_still_listed(self, page):
        """PocketTTS's official voices say only "en". The accent split Kokoro's
        bank needs drew none of them, in the list or in the character picker, so
        a Pocket installation listed its custom voices and nothing else. Drawn
        for PocketTTS -- a page drawn for another engine rightly refuses Pocket's
        answer -- and painted from a Pocket-shaped one."""
        def voice(identifier, name, official, language):
            return {"id": identifier, "display_name": name, "label": name, "engine": "pocket",
                    "official": official, "language": language, "editable": not official,
                    "deletable": not official, "compatible": True, "has_source": False}

        pocket = {"ok": True, "engine": "pocket", "default": "pocket:official:alba",
                  "voices": [voice("pocket:official:alba", "Alba", True, "en"),
                             voice("pocket:official:gb", "Brit", True, "en-GB"),
                             voice("pocket:clone:mine", "Mine", False, "")]}
        engines.select("pocket")
        pocket_list = mc_voice_ui.voices_html()
        drawn = [tree(pocket_list),
                 tree('<div id="mc-llm-chat-character-voice-list">'
                      + mc_voice_ui.character_voices_html() + '</div>')]
        engines.select("vibevoice")  # the fixture's own answers are VibeVoice's
        found = page.run("""
            await step(0);
            const list = q("[data-mc-voice-picker-list]");
            report({listed: listed(), picked: list.children.map((child) =>
                child.hasAttribute("data-mc-voice-pick")
                    ? child.getAttribute("data-mc-voice-pick")
                    : "# " + child.textContent)});
        """, trees=drawn, answers={"voice/voices": {"json": pocket},
                                    "voice/profile": {"json": {"ok": True, "engine": "pocket"}}})

        assert 'data-mc-voice-engine-id="pocket"' in pocket_list
        assert found["listed"] == ["# Official — British English", "pocket:official:gb",
                                   "# Official", "pocket:official:alba",
                                   "# Custom", "pocket:clone:mine"]
        assert found["picked"] == ["", "# British", "pocket:official:gb",
                                   "# Official", "pocket:official:alba",
                                   "# Custom", "pocket:clone:mine"]


class TestDelivery:
    def test_the_models_own_value_is_sent_as_that_and_a_moved_slider_as_its_number(
            self, page):
        found = page.run("""
            await step(0);
            const steps = q('[data-mc-voice-slider-input="steps"]');
            const first = {steps: text('[data-mc-voice-slider-value="steps"]'),
                           cfg: text('[data-mc-voice-slider-value="cfg_scale"]')};
            steps.value = "12";
            fire(steps, "input");
            const moving = text('[data-mc-voice-slider-value="steps"]');
            fire(steps, "change");
            await step(0);
            report({first, moving});
        """)

        assert found["first"] == {"steps": "model default", "cfg": "model default"}
        assert found["moving"] == "12"
        saved = [request for request in of(found, "voice/profile") if request["json"]]
        assert saved[-1]["json"] == {"profile": {"cfg_scale": None, "steps": 12}}

    def test_reset_puts_back_the_models_own_values(self, page):
        found = page.run("""
            await step(0);
            const steps = q('[data-mc-voice-slider-input="steps"]');
            steps.value = "12";
            fire(steps, "input");
            fire(steps, "change");
            await step(0);
            const reset = q("[data-mc-voice-delivery-reset]");
            fire(reset, "click");
            await step(0);
            report({});
        """)

        saved = [request for request in of(found, "voice/profile") if request["json"]]
        assert saved[-1]["json"] == {"profile": {"cfg_scale": None, "steps": None}}


# --------------------------------------------------------------------------- #
# The Voice flyout's line, and a silent stream
# --------------------------------------------------------------------------- #


FLYOUT = """
const panel = elements["mc-llm-chat-voice-engine"];
panel.offsetParent = {};
panel.fire("click", {target: panel.button});
await pump(4);
console.log(JSON.stringify(report({line: panel.line.textContent})));
"""


class TestTheFlyoutLine:
    def test_a_resident_vibevoice_says_which_card_it_is_on(self):
        answers = status_answer(engine="vibevoice", engine_state={
            "loaded": True, "state": "idle", "backend": "vibevoice",
            "where": "NVIDIA GeForce RTX 3090"})
        found = run_composer(FLYOUT, ANSWERS=json.dumps(answers))
        assert found["line"] == "\u25cf Loaded — NVIDIA GeForce RTX 3090, idle"

    def test_an_engine_that_names_no_place_is_on_the_cpu(self):
        answers = status_answer(engine_state={"loaded": True, "state": "idle"})
        found = run_composer(FLYOUT, ANSWERS=json.dumps(answers))
        assert found["line"] == "\u25cf Loaded — CPU, idle"


SILENT_STREAM = """
// The first read brings one sample and the second nothing until thirty-one
// seconds later: long enough for the page to say something about the silence.
const realFetch = globalThis.fetch;
let later = null;
globalThis.fetch = function (address, options) {
    const answer = realFetch(address, options);
    if (String(address).indexOf("tts-stream") === -1) return answer;
    return answer.then(function (response) {
        const inner = response.body.getReader();
        let reads = 0;
        const reader = {
            read() {
                reads += 1;
                if (reads === 1) return inner.read();
                return new Promise((resolve) => { later = resolve; });
            },
            cancel() { return inner.cancel(); },
        };
        response.body = {getReader: () => reader};
        return response;
    });
};
await tick(4);
turnField.value = "T1";
await pump(10);
NOW += 31000;
if (later) later({done: false, value: Uint8Array.from([0, 0])});
await tick(6);
console.log(JSON.stringify(report({later: !!later})));
"""


class TestASilentStream:
    def _run(self, engine: str) -> dict:
        answers = status_answer()
        answers["voice/tts-stream"]["chunks"] = [[0, 0]]
        if engine:
            answers["voice/tts-stream"]["headers"]["X-Model-Chain-Voice-Engine"] = engine
        return run_composer(SILENT_STREAM, ANSWERS=json.dumps(answers))

    def test_on_vibevoice_it_is_the_card_wait(self):
        found = self._run("vibevoice")
        assert found["later"] is True
        assert found["status"].startswith("VibeVoice is waiting for its turn on the graphics "
                                          "card.")

    def test_on_every_other_engine_it_is_what_it_always_was(self):
        for engine in ("kokoro", "pocket", ""):
            found = self._run(engine)
            assert found["status"].startswith("The reply is being read aloud, but no audio "
                                              "has arrived"), engine
