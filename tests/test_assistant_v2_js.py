"""The flyout's second round: read aloud, tap-to-reveal actions, Send to
prompt, Send again, and Auto Attach.

Asked for together, as "a V2 flyout menu update":

    * the read-aloud switch "does not change state when i click it", and "i
      seem to always see TTS activity in my console";
    * the Edit / Retry / Delete buttons should only show "when i tap on the
      reply", with a new SEND PROMPT there that puts the reply into the
      positive prompt, keeping LoRA tags and literal commands;
    * a message of mine left last in the thread should work the same way, with
      a SEND AGAIN;
    * an "Auto Attach" toggle in the ⋯ menu that attaches the picture showing
      in txt2img or img2img to the next message.

Each class below is one of those, run against the real browser files in the
same node harness `test_assistant_js.py` uses.
"""

from __future__ import annotations

import re

from test_assistant_js import SHELL, run

CSS = (SHELL.parent.parent / "style.css").read_text(encoding="utf-8")


def rule(selector):
    """The declarations of the one rule whose selector list names `selector`."""
    bare = re.sub(r"/\*.*?\*/", "", CSS, flags=re.DOTALL)
    found = []
    for match in re.finditer(r"([^{}]+)\{([^{}]*)\}", bare):
        names = [piece.strip() for piece in match.group(1).split(",")]
        if selector in names:
            found.append(match.group(2))
    assert found, selector + " has no rule of its own"
    return "\n".join(found)


# --------------------------------------------------------------------------- #
# Read aloud
# --------------------------------------------------------------------------- #

SWITCH = """
// A shell holding only what the read-aloud switch touches: its button, the
// status line, a store whose answer a test chooses, and Voice Chat's facade.
function switchShell(readAloud, answer) {
    const shell = Object.create(NS.Shell.prototype);
    const button = document.createElement("button");
    shell.nodes = {readAloud: button, status: {dataset: {}, textContent: ""}};
    shell.store = {
        asked: [],
        state: readAloud,
        snapshot() { return {readAloud: this.state}; },
        setReadAloud(on) {
            this.asked.push(on);
            if (answer === "refuse") return Promise.reject(new Error("refused"));
            this.state = answer === undefined ? on : answer;
            return Promise.resolve(this.state);
        },
    };
    NS.speech = {
        calls: [],
        stopPlayback(options) { this.calls.push("stop:" + options.origin); },
        setAutomaticReadAloud(on) { this.calls.push("tab:" + on); },
    };
    return shell;
}
"""


class TestTheReadAloudSwitch:
    """It started "off" whatever the setting was, never asked, and changed the
    setting by pressing Voice Chat's checkbox in a tab the flyout is usually not
    on. Nothing in the stylesheet drew its state at all."""

    def drawn(self, state):
        return run(SWITCH + """
            const shell = switchShell(%s);
            shell.renderReadAloud({readAloud: %s});
            const b = shell.nodes.readAloud;
            console.log(JSON.stringify({state: b.dataset.state,
                                        checked: b.getAttribute("aria-checked"),
                                        glyph: b.textContent, title: b.title}));
        """ % (state, state))

    def test_on_and_off_are_drawn_from_what_the_server_said(self):
        on = self.drawn("true")
        off = self.drawn("false")

        assert (on["state"], on["checked"]) == ("on", "true")
        assert (off["state"], off["checked"]) == ("off", "false")
        assert on["glyph"] != off["glyph"], "the two states share a glyph"
        assert on["title"].endswith("On") and off["title"].endswith("Off")

    def test_a_setting_nobody_has_reported_is_not_drawn_as_off(self):
        unknown = self.drawn("null")
        off = self.drawn("false")

        assert unknown["state"] == "unknown"
        assert unknown["glyph"] != off["glyph"]
        assert "Off" not in unknown["title"]

    def test_pressing_it_while_on_turns_it_off_and_silences_the_page_first(self):
        found = run(SWITCH + """
            const shell = switchShell(true);
            shell.toggleReadAloud().then(() => {
                console.log(JSON.stringify({asked: shell.store.asked,
                                            calls: NS.speech.calls,
                                            said: shell.nodes.status.textContent}));
            });
        """)

        assert found["asked"] == [False]
        assert found["calls"] == ["stop:any", "tab:false"], \
            "the speaker must stop on the press, and the tab's checkbox follow"
        assert "not" in found["said"]

    def test_pressing_it_while_off_turns_it_on_and_stops_nothing(self):
        found = run(SWITCH + """
            const shell = switchShell(false);
            shell.toggleReadAloud().then(() => {
                console.log(JSON.stringify({asked: shell.store.asked,
                                            calls: NS.speech.calls}));
            });
        """)

        assert found == {"asked": [True], "calls": ["tab:true"]}

    def test_pressing_it_before_the_server_has_said_turns_it_on(self):
        found = run(SWITCH + """
            const shell = switchShell(null);
            shell.toggleReadAloud().then(() => {
                console.log(JSON.stringify({asked: shell.store.asked}));
            });
        """)

        assert found["asked"] == [True]

    def test_the_tab_is_told_what_the_server_stored_not_what_was_asked(self):
        found = run(SWITCH + """
            const shell = switchShell(true, true);
            shell.toggleReadAloud().then(() => {
                console.log(JSON.stringify({calls: NS.speech.calls}));
            });
        """)

        assert found["calls"][-1] == "tab:true"

    def test_a_refusal_is_said_rather_than_swallowed(self):
        found = run(SWITCH + """
            const shell = switchShell(true, "refuse");
            shell.toggleReadAloud().then(() => {
                console.log(JSON.stringify({said: shell.nodes.status.textContent,
                                            kind: shell.nodes.status.dataset.kind,
                                            calls: NS.speech.calls}));
            });
        """)

        assert found["kind"] == "warn"
        assert "tab:false" not in found["calls"], \
            "the tab was told a setting the server refused"

    def test_the_stylesheet_draws_both_states(self):
        """The whole of the report: an attribute nothing drew."""
        on = rule('.forge-assistant-read-aloud[data-state="on"]')
        off = rule('.forge-assistant-read-aloud[data-state="off"]')

        assert "background" in on and "border-color" in on
        assert "opacity" not in off, "a faded switch reads as a disabled one"

    def test_every_render_draws_it(self):
        """Run through the panel's own `render`, not called directly: a switch
        drawn by a function nothing calls is the defect again."""
        found = run(SWITCH + """
            const shell = switchShell(true);
            const node = () => document.createElement("div");
            Object.assign(shell.nodes, {root: node(), unread: node(), input: node(),
                                        send: node(), stop: node()});
            ["renderSelector", "renderChip", "renderTranscript", "renderStatus",
             "applySuppression", "grow"].forEach((name) => { shell[name] = () => {}; });
            shell.canSend = () => false;
            const view = {readAloud: true, unreadTotal: 0, speech: {},
                          draft: {text: "", attachment: null}, operation: null};
            shell.render(view);
            const first = shell.nodes.readAloud.dataset.state;
            shell.render(Object.assign({}, view, {readAloud: false}));
            console.log(JSON.stringify({first,
                                        then: shell.nodes.readAloud.dataset.state}));
        """)

        assert found == {"first": "on", "then": "off"}

    def test_the_glyphs_are_the_ones_the_flyout_draws(self):
        """A speaker for on and a speaker struck through for off."""
        on = self.drawn("true")
        off = self.drawn("false")

        assert on["glyph"] == "\U0001F50A"
        assert off["glyph"] == "\U0001F507"


class TestTheStoreKnowsWhetherRepliesAreReadAloud:
    def test_the_bootstrap_says_so(self):
        found = run("""
            const store = new NS.Store();
            store.noteReadAloud({read_aloud: false});
            const off = store.snapshot().readAloud;
            store.noteReadAloud({});
            const kept = store.snapshot().readAloud;
            console.log(JSON.stringify({before: new NS.Store().snapshot().readAloud,
                                        off, kept}));
        """, sources=("store",))

        assert found == {"before": None, "off": False, "kept": False}

    def test_both_bootstraps_read_it(self):
        """The first one, and the one the panel makes whenever it looks again --
        a setting changed in LLM Studio shows the next time the panel opens."""
        store = (SHELL.parent / "forge_assistant_store.js").read_text(encoding="utf-8")
        start = store.split("Store.prototype.start = function", 1)[1] \
            .split("Store.prototype", 1)[0]
        check = store.split("Store.prototype.check = function", 1)[1] \
            .split("Store.prototype", 1)[0]

        assert "noteReadAloud(found)" in start
        assert "noteReadAloud(found)" in check

    def test_the_switch_moves_at_once_and_settles_on_the_server_s_answer(self):
        found = run("""
            const store = new NS.Store();
            store.readAloud = true;
            const seen = [];
            let asked = null;
            store.request = (path, options) => {
                asked = {path, body: JSON.parse(options.body)};
                seen.push(store.readAloud);
                return Promise.resolve({ok: true, read_aloud: false});
            };
            store.setReadAloud(false).then((stored) => {
                console.log(JSON.stringify({asked, during: seen[0], stored,
                                            after: store.readAloud}));
            });
        """, sources=("store",))

        assert found["asked"] == {"path": "/read-aloud", "body": {"read_aloud": False}}
        assert found["during"] is False
        assert found["stored"] is False and found["after"] is False

    def test_a_refused_write_puts_the_switch_back(self):
        found = run("""
            const store = new NS.Store();
            store.readAloud = true;
            store.request = () => Promise.reject(new Error("no"));
            store.setReadAloud(false).catch(() => {
                console.log(JSON.stringify({after: store.readAloud}));
            });
        """, sources=("store",))

        assert found["after"] is True


class TestAReplyAskedForWhileOffIsNeverSpoken:
    """Off means the server starts no speech for the reply at all -- no engine
    warmed, nothing synthesised -- even if writing the setting failed."""

    def stamped(self, read_aloud, action, payload="{}"):
        return run("""
            const store = new NS.Store();
            store.readAloud = %s;
            const envelope = {action: "%s", payload: %s};
            store.stampVoice(envelope);
            console.log(JSON.stringify(envelope.payload));
        """ % (read_aloud, action, payload), sources=("store",))

    def test_off_says_no_voice_on_every_command_that_asks_for_a_reply(self):
        for action in ("send", "regenerate", "continue", "resend_from_user"):
            assert self.stamped("false", action)["voice"] is False, action

    def test_on_or_unknown_leaves_it_to_the_server(self):
        assert self.stamped("true", "send")["voice"] is True
        assert self.stamped("null", "send")["voice"] is True

    def test_a_retry_is_never_restamped(self):
        """A retry is the same envelope, and the server refuses one operation id
        arriving with two different payloads."""
        assert self.stamped("false", "send", '{"voice": true}')["voice"] is True

    def test_send_stamps_generating_commands_and_nothing_else(self):
        found = run("""
            const store = new NS.Store();
            store.readAloud = false;
            store.ensureFeed = () => Promise.resolve();
            store.review = () => undefined;
            store.after = () => undefined;
            const sent = [];
            store.request = (path, options) => {
                sent.push(JSON.parse(options.body));
                return Promise.resolve({ok: true});
            };
            Promise.all([
                store.send({action: "send", payload: {text: "hi"}}),
                store.send({action: "delete_message", payload: {}}),
            ]).then(() => {
                console.log(JSON.stringify(sent.map((body) => [body.action,
                                                               body.payload.voice])));
            });
        """, sources=("store",))

        assert found == [["send", False], ["delete_message", None]]


class TestTheAnswerToAPressStaysReadable:
    """`say` is overwritten by the next render, and a press that changes the
    store causes the next render: the answer was on screen for one frame."""

    SCENARIO = """
        const shell = Object.create(NS.Shell.prototype);
        shell.nodes = {status: {dataset: {}, textContent: ""}};
        const idle = {ready: true, connected: false, idle: true, everConnected: true,
                      selection: {thread: "t"}, characters: ["Ada"], operation: null};
        shell.tell("Prompt sent to txt2img.", "info");
        shell.renderStatus(idle);
        const held = shell.nodes.status.textContent;
        shell.renderStatus(Object.assign({}, idle, {operation: {terminal: false,
                                                                status: "Generating…"}}));
        const busy = shell.nodes.status.textContent;
        shell.renderStatus(Object.assign({}, idle, {error: "Gone."}));
        const failed = shell.nodes.status.textContent;
        shell.told.at -= 60000;
        shell.renderStatus(idle);
        console.log(JSON.stringify({held, busy, failed,
                                    later: shell.nodes.status.textContent}));
    """

    def test_it_outlasts_the_idle_line_and_nothing_more(self):
        found = run(self.SCENARIO)

        assert found["held"] == "Prompt sent to txt2img."
        assert found["busy"] == "Generating…", "a reply on its way must still show"
        assert found["failed"] == "Gone.", "an error must still show"
        assert found["later"] == "Ready."
