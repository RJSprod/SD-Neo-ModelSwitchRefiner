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

from test_assistant_js import SHELL, TRANSCRIPT, run

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


# --------------------------------------------------------------------------- #
# Tap to reveal
# --------------------------------------------------------------------------- #

TAPS = """
// Real bubbles from the real `bubble()`, in a transcript the tests can walk,
// with `closest` and `contains` -- the two things the tap handler asks of the
// DOM -- answered from the parent links the stub keeps. The harness's stub
// keeps `className` and `classList` apart; a browser keeps them one, and the
// code under test reads `classList`, so here they are one as well.
const makeElement = document.createElement;
document.createElement = (tag) => {
    const node = makeElement(tag);
    let name = "";
    Object.defineProperty(node, "className", {
        get() { return name; },
        set(value) {
            name = String(value || "");
            name.split(/\\s+/).filter(Boolean).forEach((one) => node.classList.add(one));
        },
    });
    return node;
};
function tapShell(roles) {
    const shell = Object.create(NS.Shell.prototype);
    shell.settings = {bubbleWidth: 80};
    const transcript = document.createElement("div");
    shell.nodes = {transcript, status: {dataset: {}, textContent: ""}};
    const messages = roles.map((role, index) => ({index, role, text: "m" + index,
                                                   active: 0, versions: ["m" + index]}));
    const view = {conversation: {conversation: {revision: 7}, messages}};
    const link = (parent, child) => { child.parentNode = parent; return child; };
    const within = (node, target) => {
        for (let at = target; at; at = at.parentNode) if (at === node) return true;
        return false;
    };
    messages.forEach((row) => {
        const node = shell.bubble(row, view);
        node.dataset.key = "k" + row.index;
        link(transcript, node);
        node.children.forEach((child) => {
            link(node, child);
            (child.children || []).forEach((grand) => link(child, grand));
        });
        node.contains = (target) => within(node, target);
        transcript.children.push(node);
    });
    const tagOf = (node) => String(node.tagName || "").toLowerCase();
    [transcript].concat(transcript.children).forEach((node) => {
        const all = [node].concat(node.children || []);
        all.forEach((item) => {
            item.closest = (selector) => {
                for (let at = item; at; at = at.parentNode) {
                    const names = at.classList ? Array.from(at.classList.names) : [];
                    const wanted = selector.split(",").map((part) => part.trim());
                    if (wanted.some((one) => (one.startsWith(".")
                        ? names.indexOf(one.slice(1)) >= 0 : one === tagOf(at)))) return at;
                }
                return null;
            };
            (item.children || []).forEach((grand) => { grand.closest = item.closest.bind(grand); });
        });
    });
    return {shell, transcript, view};
}
const openRows = (transcript) => transcript.children
    .filter((node) => node.dataset.actions)
    .map((node) => [node.dataset.key,
                    node.children.find((c) => c.classList.contains("forge-assistant-actions"))
                        .hidden === false,
                    node.getAttribute("aria-expanded")]);
const textOf = (node) => node.children[0];
"""


class TestTheActionsWaitForATap:
    """"i want these buttons to only show up when i tap on the reply."""

    def test_nothing_is_drawn_until_a_message_is_tapped(self):
        found = run(TAPS + """
            const {transcript} = tapShell(["user", "assistant", "user", "assistant"]);
            console.log(JSON.stringify({rows: openRows(transcript),
                                        tappable: transcript.children.map(
                                            (node) => node.getAttribute("tabindex"))}));
        """)

        # Older reply (Send to prompt) and the newest reply are tappable; an
        # older message of yours has nothing to offer and is not.
        assert found["tappable"] == [None, "0", None, "0"]
        assert all(shown is False for _, shown, _ in found["rows"])
        assert all(expanded == "false" for _, _, expanded in found["rows"])

    def test_a_tap_shows_that_message_s_actions_and_a_second_hides_them(self):
        found = run(TAPS + """
            const {shell, transcript} = tapShell(["user", "assistant"]);
            const reply = transcript.children[1];
            shell.tapBubble({target: textOf(reply)});
            const opened = openRows(transcript);
            shell.tapBubble({target: textOf(reply)});
            console.log(JSON.stringify({opened, closed: openRows(transcript),
                                        revealed: reply.classList.contains(
                                            "forge-assistant-revealed")}));
        """)

        assert found["opened"] == [["k1", True, "true"]]
        assert found["closed"] == [["k1", False, "false"]]
        assert found["revealed"] is False

    def test_only_one_message_is_open_at_a_time(self):
        found = run(TAPS + """
            const {shell, transcript} = tapShell(["assistant", "user", "assistant"]);
            const [older, , newest] = transcript.children;
            shell.tapBubble({target: textOf(older)});
            // The press that opens the newest lands first on the window.
            shell.dismissActions({target: textOf(newest)});
            shell.tapBubble({target: textOf(newest)});
            console.log(JSON.stringify(openRows(transcript)));
        """)

        assert found == [["k0", False, "false"], ["k2", True, "true"]]

    def test_a_press_anywhere_else_puts_them_away(self):
        found = run(TAPS + """
            const {shell, transcript} = tapShell(["assistant"]);
            shell.tapBubble({target: textOf(transcript.children[0])});
            const kept = shell.dismissActions({target: textOf(transcript.children[0])});
            const away = shell.dismissActions({target: document.createElement("div")});
            console.log(JSON.stringify({kept, away, rows: openRows(transcript)}));
        """)

        assert found["kept"] is False, "a press on the open message closed it early"
        assert found["away"] is True
        assert found["rows"] == [["k0", False, "false"]]

    def test_a_press_on_a_link_or_a_button_is_that_control_s(self):
        found = run(TAPS + """
            const {shell, transcript} = tapShell(["assistant"]);
            const reply = transcript.children[0];
            const anchor = document.createElement("a");
            anchor.parentNode = textOf(reply);
            anchor.closest = (selector) => selector.indexOf("a") >= 0 ? anchor : null;
            const viaLink = shell.tapBubble({target: anchor});
            console.log(JSON.stringify({viaLink, rows: openRows(transcript)}));
        """)

        assert found == {"viaLink": False, "rows": [["k0", False, "false"]]}

    def test_selecting_text_is_reading_not_tapping(self):
        found = run(TAPS + """
            const {shell, transcript} = tapShell(["assistant"]);
            const reply = transcript.children[0];
            globalThis.getSelection = () => ({isCollapsed: false, anchorNode: textOf(reply)});
            const toggled = shell.tapBubble({target: textOf(reply)});
            console.log(JSON.stringify({toggled, rows: openRows(transcript)}));
        """)

        assert found == {"toggled": False, "rows": [["k0", False, "false"]]}

    def test_the_keyboard_can_do_what_a_tap_does(self):
        found = run(TAPS + """
            const {shell, transcript} = tapShell(["assistant"]);
            const reply = transcript.children[0];
            let prevented = 0;
            const key = (name) => ({key: name, target: reply,
                                    preventDefault() { prevented += 1; }});
            shell.bubbleKey(key("a"));
            const ignored = openRows(transcript);
            shell.bubbleKey(key("Enter"));
            const entered = openRows(transcript);
            shell.bubbleKey(key(" "));
            console.log(JSON.stringify({ignored, entered, spaced: openRows(transcript),
                                        prevented}));
        """)

        assert found["ignored"] == [["k0", False, "false"]]
        assert found["entered"] == [["k0", True, "true"]]
        assert found["spaced"] == [["k0", False, "false"]]
        assert found["prevented"] == 2

    def test_pressing_an_action_puts_the_row_away(self):
        found = run(TAPS + """
            const {shell, transcript} = tapShell(["user", "assistant"]);
            const reply = transcript.children[1];
            const acted = [];
            shell.act = (action) => acted.push(action);
            shell.tapBubble({target: textOf(reply)});
            const bar = reply.children.find((c) => c.classList.contains("forge-assistant-actions"));
            bar.children[bar.children.length - 1].handlers.click.forEach((fn) => fn());
            console.log(JSON.stringify({acted, rows: openRows(transcript)}));
        """)

        assert found["acted"] == ["send_prompt"]
        assert found["rows"] == [["k1", False, "false"]]

    def test_the_open_row_survives_a_redraw_and_goes_with_its_message(self):
        """A reply streaming in redraws the transcript once a frame; a row that
        closed on every word could not be used while a reply was arriving."""
        found = run(TAPS + """
            const {shell, transcript} = tapShell(["user", "assistant"]);
            shell.tapBubble({target: textOf(transcript.children[1])});
            // A redraw that kept the node: the key is still on the page.
            shell.reveal(shell.revealed);
            const kept = openRows(transcript);
            // A redraw after the thread moved: the key is gone.
            transcript.children[1].dataset.key = "k1-after";
            shell.reveal(shell.revealed);
            console.log(JSON.stringify({kept, after: openRows(transcript),
                                        remembered: shell.revealed}));
        """)

        assert found["kept"] == [["k1", True, "true"]]
        assert found["after"] == [["k1-after", False, "false"]]
        assert found["remembered"] == ""

    def test_a_message_drawn_again_comes_back_open(self):
        """Through the real `renderTranscript`: a snapshot that arrives without
        its conversation empties the transcript, and when the conversation
        comes back its bubbles are new nodes. The row somebody had open is
        opened again on the new one."""
        found = run(TAPS + TRANSCRIPT + """
            const transcript = fakeTranscript();
            const shell = shellWith(transcript);
            shell.bubble = NS.Shell.prototype.bubble;
            shell.updateBubble = () => undefined;
            const messages = [{index: 0, role: "user", text: "q", active: 0, versions: ["q"]},
                              {index: 1, role: "assistant", text: "a", active: 0,
                               versions: ["a"]}];
            const view = {conversation: {conversation: {revision: 3}, messages},
                          selection: {epoch: "e1"}, unread: 0, operation: null};
            shell.renderTranscript(view);
            const reply = transcript.children[1];
            shell.reveal(reply.dataset.key);
            shell.renderTranscript({conversation: null, selection: {epoch: "e1"}});
            shell.renderTranscript(view);
            const again = transcript.children[1];
            const bar = again.children.find((c) => c.classList.contains("forge-assistant-actions"));
            console.log(JSON.stringify({rebuilt: again !== reply, open: bar.hidden === false}));
        """)

        assert found == {"rebuilt": True, "open": True}

    def test_a_reply_still_being_written_offers_nothing(self):
        found = run(TAPS + """
            const shell = Object.create(NS.Shell.prototype);
            shell.settings = {bubbleWidth: 80};
            const row = {index: 1, role: "assistant", text: "half a", active: 0,
                         versions: ["half a"], provisional: true};
            const view = {conversation: {conversation: {revision: 1},
                                         messages: [{index: 0, role: "user", text: "q"}]}};
            const node = shell.bubble(row, view);
            console.log(JSON.stringify({actions: node.dataset.actions || null,
                                        tabindex: node.getAttribute("tabindex")}));
        """)

        assert found == {"actions": None, "tabindex": None}

    def test_the_stylesheet_marks_what_can_be_tapped_and_what_is_open(self):
        assert "cursor: pointer" in rule(".forge-assistant-bubble[data-actions]")
        assert "border-color" in rule(".forge-assistant-revealed")

    def test_the_transcript_listens_for_the_tap(self):
        """Delegated from the transcript: bubbles are rebuilt whenever the
        thread moves, and a listener bound to each would be lost with it."""
        shell = SHELL.read_text(encoding="utf-8")
        wire = shell.split("Shell.prototype.wire = function", 1)[1] \
            .split("Shell.prototype.grow", 1)[0]

        assert 'this.on(nodes.transcript, "click", (event) => this.tapBubble(event));' in wire
        assert 'this.on(window, "pointerdown", (event) => this.dismissActions(event), true);' \
            in wire


# --------------------------------------------------------------------------- #
# Send to prompt
# --------------------------------------------------------------------------- #


def prompt_from(reply, current):
    return run("""
        console.log(JSON.stringify(NS.promptFrom(%s, %s)));
    """ % (json_string(reply), json_string(current)))


def json_string(text):
    import json

    return json.dumps(text)


class TestWhatSendToPromptKeeps:
    """"remove everything not in literal or lora, and prepend the LLM prompt
    following by a new line for separation." """

    def test_the_reply_replaces_the_prose_and_the_tags_follow_it(self):
        found = prompt_from("a lighthouse at dusk, volumetric fog",
                            "portrait of a woman, <lora:detail:0.5> blue hat")

        assert found == "a lighthouse at dusk, volumetric fog\n<lora:detail:0.5>"

    def test_every_literal_command_is_kept_exactly_as_typed(self):
        found = prompt_from("new scene",
                            "[[<lora:realfilter:1>]] old words +[[ A ]] more -[[__light__]]")

        assert found == "new scene\n[[<lora:realfilter:1>]] +[[ A ]] -[[__light__]]"

    def test_what_is_kept_keeps_its_order(self):
        found = prompt_from("x", "-[[D]] <lora:b:1> words +[[A]] <LyCo:c:0.3> [[B]]")

        assert found == "x\n-[[D]] <lora:b:1> +[[A]] <LyCo:c:0.3> [[B]]"

    def test_a_tag_inside_a_literal_is_kept_once(self):
        found = prompt_from("x", "[[<lora:inside:1>]] and <hypernet:net:1>")

        assert found == "x\n[[<lora:inside:1>]] <hypernet:net:1>"

    def test_only_the_three_extra_network_kinds_are_tags(self):
        """The closed list extra_networks.py protects, for its reason: an open
        `<word:...>` shape would keep somebody else's syntax by accident."""
        found = prompt_from("x", "<lora:a:1> <embedding:b> <wildcard:c> <HYPERNET:d:1>")

        assert found == "x\n<lora:a:1> <HYPERNET:d:1>"

    def test_an_empty_or_unclosed_literal_is_not_a_literal(self):
        """The grammar's own answers: an empty command carries nothing, and one
        never closed is ordinary text -- so it goes, as text does. A LoRA tag
        in that text is still a LoRA tag, and stays."""
        found = prompt_from("x", "words [[]] more [[never closed <lora:after:1>")

        assert found == "x\n<lora:after:1>"

    def test_the_first_close_closes(self):
        found = prompt_from("x", "[[a [[b]] c]] <lora:z:1>")

        assert found == "x\n[[a [[b]] <lora:z:1>"

    def test_with_nothing_to_keep_it_is_just_the_reply(self):
        found = prompt_from("  a prompt with room around it \n", "all prose, nothing kept")

        assert found == "a prompt with room around it"

    def test_the_reply_goes_in_as_written(self):
        """Guessing which of a reply's sentences is "the prompt" is how a
        sentence somebody wanted disappears."""
        reply = "Here you go:\n\nA **misty** harbour, 35mm"
        found = prompt_from(reply, "")

        assert found == reply

    def test_a_keeper_is_never_mistaken_for_a_plus_sign_in_prose(self):
        """Only a sign *immediately* before `[[` is a sign -- the grammar's
        rule, which is why nobody has to escape arithmetic."""
        found = prompt_from("x", "2 + [[B]] and 3 -[[C]]")

        assert found == "x\n[[B]] -[[C]]"


PROMPTS = """
// The two prompt boxes Forge draws, as Gradio draws them: a wrapper carrying
// the id, and a textarea inside it.
const typed = [];
function promptBoxes() {
    const boxes = {};
    ["txt2img_prompt", "img2img_prompt"].forEach((id) => {
        const holder = document.createElement("div");
        holder.id = id;
        const area = document.createElement("textarea");
        area.tagName = "TEXTAREA";
        area.value = "old words <lora:keep:1>";
        area.dispatchEvent = (event) => { typed.push(id + ":" + event.type); return true; };
        holder.querySelector = (selector) => selector === "textarea" ? area : null;
        boxes[id] = area;
    });
    const byId = Object.assign({}, ...Object.keys(boxes).map((id) => ({[id]: {
        tagName: "DIV", querySelector: (s) => s === "textarea" ? boxes[id] : null}})));
    const previous = document.getElementById;
    document.getElementById = (id) => byId[id] || previous(id);
    return boxes;
}
function promptShell(workspace) {
    const shell = Object.create(NS.Shell.prototype);
    shell.nodes = {status: {dataset: {}, textContent: ""}};
    shell.host = {getActiveWorkspace: () => workspace};
    return shell;
}
const reply = {index: 3, role: "assistant", text: "a quiet harbour", active: 0};
"""


class TestWhereSendToPromptWrites:
    def send(self, workspace):
        return run(PROMPTS + """
            const boxes = promptBoxes();
            const shell = promptShell("%s");
            const done = shell.sendToPrompt(reply);
            console.log(JSON.stringify({done, typed,
                                        t2i: boxes.txt2img_prompt.value,
                                        i2i: boxes.img2img_prompt.value,
                                        said: shell.nodes.status.textContent}));
        """ % workspace)

    def test_on_txt2img_it_writes_txt2img_s_prompt(self):
        found = self.send("tab_txt2img")

        assert found["t2i"] == "a quiet harbour\n<lora:keep:1>"
        assert found["i2i"] == "old words <lora:keep:1>"
        assert found["said"] == "Prompt sent to txt2img."

    def test_on_img2img_it_writes_img2img_s_prompt(self):
        found = self.send("tab_img2img")

        assert found["i2i"] == "a quiet harbour\n<lora:keep:1>"
        assert found["t2i"] == "old words <lora:keep:1>"
        assert found["said"] == "Prompt sent to img2img."

    def test_from_anywhere_else_it_writes_txt2img_s(self):
        found = self.send("tab_llm_studio")

        assert found["t2i"] == "a quiet harbour\n<lora:keep:1>"
        assert found["said"] == "Prompt sent to txt2img."

    def test_gradio_is_told_and_nothing_is_pressed(self):
        """Typing, as far as the page can tell: an input event, which is what
        Gradio stores a textbox's value from. No click on anything -- "it
        should not cause an image to be auto generated"."""
        found = self.send("tab_txt2img")

        assert found["typed"] == ["txt2img_prompt:input"]

    def test_forge_s_own_helper_is_used_where_there_is_one(self):
        found = run(PROMPTS + """
            const boxes = promptBoxes();
            const helped = [];
            globalThis.updateInput = (element) => helped.push(element === boxes.txt2img_prompt);
            promptShell("tab_txt2img").sendToPrompt(reply);
            console.log(JSON.stringify({helped, typed}));
        """)

        assert found == {"helped": [True], "typed": []}

    def test_a_page_without_the_prompt_says_so(self):
        found = run(PROMPTS + """
            const shell = promptShell("tab_txt2img");
            const done = shell.sendToPrompt(reply);
            console.log(JSON.stringify({done, said: shell.nodes.status.textContent,
                                        kind: shell.nodes.status.dataset.kind}));
        """)

        assert found["done"] is False
        assert found["kind"] == "warn"
        assert "txt2img" in found["said"]

    def test_the_action_is_the_panel_s_own_and_sends_no_command(self):
        found = run(PROMPTS + """
            promptBoxes();
            const shell = promptShell("tab_txt2img");
            const sent = [];
            shell.store = {envelope: () => ({}), send: (e) => { sent.push(e); return Promise.resolve({ok: true}); }};
            shell.act("send_prompt", reply, 7);
            console.log(JSON.stringify({sent: sent.length,
                                        said: shell.nodes.status.textContent}));
        """)

        assert found == {"sent": 0, "said": "Prompt sent to txt2img."}


class TestSendAgain:
    def test_it_asks_the_server_to_answer_that_message(self):
        found = run("""
            const shell = Object.create(NS.Shell.prototype);
            shell.nodes = {status: {dataset: {}, textContent: ""}};
            const store = new NS.Store();
            store.selection = {character: "Ada", thread: "t", epoch: "x"};
            const sent = [];
            store.send = (envelope) => { sent.push(envelope); return Promise.resolve({ok: true}); };
            shell.store = store;
            shell.act("resend_from_user", {index: 4, role: "user", text: "ask", active: 0}, 9);
            const envelope = sent[0];
            console.log(JSON.stringify({action: envelope.action, target: envelope.target,
                                        expected: envelope.expected_revision}));
        """, sources=("shell", "store"))

        assert found == {"action": "resend_from_user",
                         "target": {"index": 4, "version": 0},
                         "expected": {"kind": "revision", "value": 9}}

    def test_it_is_a_command_that_opens_the_feed_for_its_reply(self):
        """The store's list of commands that anticipate a reply: without it,
        Send again's answer would arrive on no feed at all."""
        store = (SHELL.parent / "forge_assistant_store.js").read_text(encoding="utf-8")

        assert re.search(r'const GENERATING = \[[^\]]*"resend_from_user"', store)


# --------------------------------------------------------------------------- #
# Auto Attach
# --------------------------------------------------------------------------- #

GALLERY = """
// Forge's galleries as Gradio 4.40 draws them, reduced to what is asked of
// them: `querySelector` by the selectors Auto Attach tries, in order, and
// `querySelectorAll("img")` for the last resort.
function picture(src, live) {
    const image = document.createElement("img");
    image.src = src;
    image.closest = (selector) => (live && selector === ".livePreview") ? {} : null;
    return image;
}
function gallery(id, bySelector, all) {
    const node = {tagName: "DIV",
                  querySelector: (selector) => bySelector[selector] || null,
                  querySelectorAll: (selector) => selector === "img" ? (all || []) : []};
    const previous = document.getElementById;
    document.getElementById = (wanted) => wanted === id ? node : previous(wanted);
    return node;
}
const PREVIEW = ".preview img[data-testid=\\"detailed-image\\"]";
const SELECTED = ".thumbnail-item.selected img";
const FIRST = ".thumbnail-item img";
globalThis.localStorage = {
    store: {},
    getItem(key) { return this.store[key] === undefined ? null : this.store[key]; },
    setItem(key, value) { this.store[key] = String(value); },
};
"""


class TestWhichPictureIsShowing:
    """Forge's own answer, the one its Send to img2img buttons use: the picture
    you clicked, and the first when you have not clicked one."""

    def showing(self, setup, workspace="tab_txt2img"):
        return run(GALLERY + setup + """
            console.log(JSON.stringify(NS.showingPicture("%s")));
        """ % workspace)

    def test_the_picture_open_in_the_preview_wins(self):
        found = self.showing("""
            gallery("txt2img_gallery", {[PREVIEW]: picture("/file=/out/2.png"),
                                        [FIRST]: picture("/file=/out/1.png")});
        """)

        assert found == {"src": "/file=/out/2.png", "label": "txt2img", "name": "2.png"}

    def test_then_the_selected_thumbnail_then_the_first(self):
        selected = self.showing("""
            gallery("txt2img_gallery", {[SELECTED]: picture("/file=/out/3.png"),
                                        [FIRST]: picture("/file=/out/1.png")});
        """)
        first = self.showing("""
            gallery("txt2img_gallery", {[FIRST]: picture("/file=/out/1.png")});
        """)

        assert selected["src"] == "/file=/out/3.png"
        assert first["src"] == "/file=/out/1.png"

    def test_the_live_preview_of_a_running_generation_is_not_the_picture(self):
        found = self.showing("""
            gallery("txt2img_gallery", {},
                    [picture("/live.png", true), picture("/file=/out/done.png")]);
        """)

        assert found["src"] == "/file=/out/done.png"

    def test_an_empty_gallery_has_nothing_showing(self):
        assert self.showing('gallery("txt2img_gallery", {}, []);') is None

    def test_img2img_reads_its_own_gallery(self):
        found = self.showing("""
            gallery("txt2img_gallery", {[FIRST]: picture("/file=/out/t.png")});
            gallery("img2img_gallery", {[FIRST]: picture("/file=/out/i.png")});
        """, workspace="tab_img2img")

        assert found == {"src": "/file=/out/i.png", "label": "img2img", "name": "i.png"}

    def test_any_other_workspace_has_nothing_to_attach(self):
        found = self.showing("""
            gallery("txt2img_gallery", {[FIRST]: picture("/file=/out/t.png")});
        """, workspace="tab_extras")

        assert found is None

    def test_a_file_name_is_read_out_of_gradio_s_address(self):
        found = run("""
            console.log(JSON.stringify([
                NS.fileNameOf("https://forge:7860/file=/tmp/gradio/ab12/00012-123.png"),
                NS.fileNameOf("/file=C%3A%5Cout%5Cgrid%20one.webp?t=5"),
                NS.fileNameOf(""),
            ]));
        """)

        assert found == ["00012-123.png", "C:\\out\\grid one.webp", "image.png"]


AUTO = GALLERY + """
function autoShell(options) {
    options = options || {};
    const shell = Object.create(NS.Shell.prototype);
    shell.state = {autoAttach: options.on !== false};
    shell.autoSent = new Map();
    shell.nodes = {status: {dataset: {}, textContent: ""},
                   attach: document.createElement("button"),
                   input: {value: options.text || "what do you think?"}};
    shell.host = {getActiveWorkspace: () => options.workspace || "tab_txt2img"};
    const drafts = {};
    const store = {
        selection: {character: "Ada", thread: "t1", epoch: "e1"},
        submitted: [],
        uploaded: [],
        capabilities: options.capabilities || {vision: true},
        messages: options.messages || [],
        draft(key) {
            const at = key || NS.conversationKey(this.selection.character, this.selection.thread);
            return drafts[at] || (drafts[at] = {text: "", attachment: null});
        },
        snapshot() {
            const draft = this.draft();
            return {ready: true, selection: Object.assign({}, this.selection),
                    capabilities: this.capabilities, draft,
                    conversation: {messages: this.messages}, operation: null};
        },
        setDraftText(text) { this.draft().text = text; },
        setAttachment(attachment, key) { this.draft(key).attachment = attachment; },
        upload(file) {
            this.uploaded.push({name: file.name, type: file.type});
            if (options.refuseUpload) return Promise.reject(new Error("too large"));
            return Promise.resolve({token: "tok-" + this.uploaded.length, name: file.name});
        },
        submit() {
            const draft = this.draft();
            this.submitted.push({text: draft.text, attachment: draft.attachment
                ? {token: draft.attachment.token, from: draft.attachment.from} : null});
            return Promise.resolve({ok: true});
        },
    };
    shell.store = store;
    shell.canSend = () => true;
    globalThis.fetch = (url) => {
        store.fetched = url;
        if (options.fetchFails) return Promise.resolve({ok: false, status: 404});
        return Promise.resolve({ok: true,
                                blob: () => Promise.resolve(new Blob(["png"], {type: "image/png"}))});
    };
    return shell;
}
const settle = () => new Promise((resolve) => setImmediate(resolve));
"""


def run_auto(scenario):
    """The shell with the store beside it: conversation keys and the base
    path the preference is stored under are the store's."""
    return run(scenario, sources=("shell", "store"))


class TestAutoAttachSends:
    def test_the_showing_picture_goes_with_the_message(self):
        found = run_auto(AUTO + """
            gallery("txt2img_gallery", {[FIRST]: picture("/file=/out/7.png")});
            const shell = autoShell();
            shell.send();
            settle().then(settle).then(settle).then(() => {
                console.log(JSON.stringify({fetched: shell.store.fetched,
                                            uploaded: shell.store.uploaded,
                                            submitted: shell.store.submitted,
                                            remembered: shell.autoSent.get(
                                                NS.conversationKey("Ada", "t1"))}));
            });
        """)

        assert found["fetched"] == "/file=/out/7.png"
        assert found["uploaded"] == [{"name": "7.png", "type": "image/png"}]
        assert found["submitted"] == [{"text": "what do you think?",
                                       "attachment": {"token": "tok-1", "from": "txt2img"}}]
        assert found["remembered"] == "/file=/out/7.png"

    def test_off_attaches_nothing(self):
        found = run_auto(AUTO + """
            gallery("txt2img_gallery", {[FIRST]: picture("/file=/out/7.png")});
            const shell = autoShell({on: false});
            shell.send();
            settle().then(settle).then(() => {
                console.log(JSON.stringify({fetched: shell.store.fetched || null,
                                            submitted: shell.store.submitted}));
            });
        """)

        assert found == {"fetched": None,
                         "submitted": [{"text": "what do you think?", "attachment": None}]}

    def test_not_on_an_image_tab_attaches_nothing(self):
        found = run_auto(AUTO + """
            gallery("txt2img_gallery", {[FIRST]: picture("/file=/out/7.png")});
            const shell = autoShell({workspace: "tab_llm_studio"});
            shell.send();
            settle().then(settle).then(() => {
                console.log(JSON.stringify({fetched: shell.store.fetched || null,
                                            attachment: shell.store.submitted[0].attachment}));
            });
        """)

        assert found == {"fetched": None, "attachment": None}

    def test_an_empty_gallery_attaches_nothing(self):
        found = run_auto(AUTO + """
            gallery("txt2img_gallery", {}, []);
            const shell = autoShell();
            shell.send();
            settle().then(settle).then(() => {
                console.log(JSON.stringify({attachment: shell.store.submitted[0].attachment}));
            });
        """)

        assert found == {"attachment": None}

    def test_a_picture_attached_by_hand_is_the_one_sent(self):
        found = run_auto(AUTO + """
            gallery("txt2img_gallery", {[FIRST]: picture("/file=/out/7.png")});
            const shell = autoShell();
            shell.store.draft().attachment = {localId: "mine", state: "ready", token: "hand"};
            shell.send();
            settle().then(settle).then(() => {
                console.log(JSON.stringify({fetched: shell.store.fetched || null,
                                            attachment: shell.store.submitted[0].attachment}));
            });
        """)

        assert found == {"fetched": None, "attachment": {"token": "hand"}}

    def test_a_model_that_cannot_see_is_sent_the_words_and_you_are_told(self):
        found = run_auto(AUTO + """
            gallery("txt2img_gallery", {[FIRST]: picture("/file=/out/7.png")});
            const shell = autoShell({capabilities: {vision: false}});
            shell.send();
            settle().then(settle).then(() => {
                console.log(JSON.stringify({fetched: shell.store.fetched || null,
                                            submitted: shell.store.submitted.length,
                                            said: shell.nodes.status.textContent,
                                            kind: shell.nodes.status.dataset.kind}));
            });
        """)

        assert found["fetched"] is None
        assert found["submitted"] == 1
        assert found["kind"] == "warn" and "cannot see" in found["said"]

    def test_the_same_picture_is_not_sent_twice_while_the_thread_still_has_one(self):
        found = run_auto(AUTO + """
            gallery("txt2img_gallery", {[FIRST]: picture("/file=/out/7.png")});
            const shell = autoShell({messages: [{role: "user", attachment: {name: "7.png"}}]});
            shell.autoSent.set(NS.conversationKey("Ada", "t1"), "/file=/out/7.png");
            shell.send();
            settle().then(settle).then(() => {
                console.log(JSON.stringify({fetched: shell.store.fetched || null,
                                            attachment: shell.store.submitted[0].attachment,
                                            said: shell.nodes.status.textContent}));
            });
        """)

        assert found["fetched"] is None
        assert found["attachment"] is None
        assert "Same picture" in found["said"]

    def test_the_same_picture_is_sent_again_once_the_thread_has_none(self):
        """The message that carried it was deleted: the model no longer has it."""
        found = run_auto(AUTO + """
            gallery("txt2img_gallery", {[FIRST]: picture("/file=/out/7.png")});
            const shell = autoShell({messages: [{role: "user", text: "no picture here"}]});
            shell.autoSent.set(NS.conversationKey("Ada", "t1"), "/file=/out/7.png");
            shell.send();
            settle().then(settle).then(settle).then(() => {
                console.log(JSON.stringify(shell.store.submitted[0].attachment));
            });
        """)

        assert found == {"token": "tok-1", "from": "txt2img"}

    def test_a_new_picture_in_the_gallery_is_sent(self):
        found = run_auto(AUTO + """
            gallery("txt2img_gallery", {[FIRST]: picture("/file=/out/8.png")});
            const shell = autoShell({messages: [{role: "user", attachment: {name: "7.png"}}]});
            shell.autoSent.set(NS.conversationKey("Ada", "t1"), "/file=/out/7.png");
            shell.send();
            settle().then(settle).then(settle).then(() => {
                console.log(JSON.stringify(shell.store.submitted[0].attachment));
            });
        """)

        assert found == {"token": "tok-1", "from": "txt2img"}

    @staticmethod
    def failing(option):
        return run_auto(AUTO + """
            gallery("txt2img_gallery", {[FIRST]: picture("/file=/out/7.png")});
            const shell = autoShell({%s: true});
            shell.send();
            settle().then(settle).then(settle).then(() => {
                console.log(JSON.stringify({submitted: shell.store.submitted,
                                            said: shell.nodes.status.textContent,
                                            kind: shell.nodes.status.dataset.kind,
                                            remembered: shell.autoSent.size}));
            });
        """ % option)

    def test_a_picture_that_cannot_be_read_is_left_out_and_said(self):
        found = self.failing("fetchFails")

        assert found["submitted"] == [{"text": "what do you think?", "attachment": None}]
        assert found["kind"] == "warn" and "Sent without it" in found["said"]
        assert found["remembered"] == 0

    def test_a_picture_the_server_refuses_is_left_out_and_said(self):
        found = self.failing("refuseUpload")

        assert found["submitted"] == [{"text": "what do you think?", "attachment": None}]
        assert "too large" in found["said"]

    def test_a_conversation_changed_mid_upload_is_not_sent_to(self):
        found = run_auto(AUTO + """
            gallery("txt2img_gallery", {[FIRST]: picture("/file=/out/7.png")});
            const shell = autoShell();
            const upload = shell.store.upload.bind(shell.store);
            shell.store.upload = (file) => {
                shell.store.selection = {character: "Ada", thread: "t2", epoch: "e2"};
                return upload(file);
            };
            shell.send();
            settle().then(settle).then(settle).then(() => {
                const kept = shell.store.draft(NS.conversationKey("Ada", "t1")).attachment;
                console.log(JSON.stringify({submitted: shell.store.submitted.length,
                                            kept: kept && kept.token,
                                            kind: shell.nodes.status.dataset.kind}));
            });
        """)

        assert found == {"submitted": 0, "kept": "tok-1", "kind": "warn"}


class TestTheAutoAttachSwitch:
    def test_it_is_a_mode_in_the_menu_that_reports_its_state(self):
        found = run_auto(AUTO + """
            const shell = autoShell({on: false});
            shell.closeMenu = () => undefined;
            const item = shell.autoAttachItem();
            const before = {role: item.getAttribute("role"),
                            checked: item.getAttribute("aria-checked"),
                            label: item.textContent};
            item.handlers.click.forEach((fn) => fn());
            console.log(JSON.stringify({before, on: shell.state.autoAttach,
                                        again: shell.autoAttachItem().getAttribute("aria-checked"),
                                        paperclip: shell.nodes.attach.dataset.auto,
                                        said: shell.nodes.status.textContent}));
        """)

        assert found["before"] == {"role": "menuitemcheckbox", "checked": "false",
                                   "label": "Auto Attach"}
        assert found["on"] is True and found["again"] == "true"
        assert found["paperclip"] == "on"
        assert "Auto Attach is on" in found["said"]

    def test_it_is_remembered_in_this_browser(self):
        found = run_auto(AUTO + """
            const shell = autoShell({on: false});
            shell.setAutoAttach(true);
            const fresh = Object.create(NS.Shell.prototype);
            fresh.state = {autoAttach: false};
            fresh._restoreAutoAttach();
            console.log(JSON.stringify({restored: fresh.state.autoAttach,
                                        stored: Object.values(localStorage.store)}));
        """)

        assert found == {"restored": True, "stored": ["on"]}

    def test_storage_that_throws_leaves_it_off_and_working(self):
        found = run_auto(AUTO + """
            globalThis.localStorage = {getItem() { throw new Error("blocked"); },
                                       setItem() { throw new Error("blocked"); }};
            const fresh = Object.create(NS.Shell.prototype);
            fresh.state = {};
            fresh._restoreAutoAttach();
            const shell = autoShell({on: false});
            shell.setAutoAttach(true);
            console.log(JSON.stringify({restored: fresh.state.autoAttach,
                                        on: shell.state.autoAttach}));
        """)

        assert found == {"restored": False, "on": True}

    def test_the_menu_offers_it_beside_free_float(self):
        found = run_auto(AUTO + """
            const shell = autoShell();
            shell.closeMenu = () => undefined;
            shell.host.listUtilities = () => [];
            console.log(JSON.stringify(shell.utilityItems().map((item) => item.textContent)));
        """)

        assert found[:2] == ["Free Float", "Auto Attach"]

    def test_the_paperclip_is_lit_while_it_is_on(self):
        on = rule('.forge-assistant-attach[data-auto="on"]')

        assert "background" in on and "border-color" in on

    def test_the_chip_says_where_an_automatic_picture_came_from(self):
        found = run_auto(AUTO + """
            const shell = autoShell();
            shell.nodes.chip = document.createElement("div");
            shell.renderChip({localId: "a", state: "uploading", name: "7.png",
                              preview: "/file=/out/7.png", from: "img2img"});
            const said = shell.nodes.chip.children.find(
                (child) => child.className === "forge-assistant-chip-state");
            console.log(JSON.stringify(said.textContent));
        """)

        assert found == "From img2img \u00b7 Uploading…"


class TestAPictureTheServerWouldRefuse:
    """Forge can be told to save in a format staging does not read, and an
    upscale can pass its size limit. Those are drawn onto a canvas and sent as
    a JPEG; anything staging takes is sent exactly as it is."""

    SCENARIO = """
        globalThis.createImageBitmap = (blob) => Promise.resolve({width: 4096, height: 2048});
        const drawn = [];
        const makeElement = document.createElement;
        document.createElement = (tag) => {
            const node = makeElement(tag);
            if (tag === "canvas") {
                node.getContext = () => ({drawImage: (b, x, y, w, h) => drawn.push([w, h])});
                node.toBlob = (done, type) => done(new Blob(["jpg"], {type}));
            }
            return node;
        };
        const check = (blob) => NS.stageable(blob).then((out) => ({
            same: out === blob, type: out.type, drawn: drawn.slice()}));
        Promise.all([
            check(new Blob(["png"], {type: "image/png"})),
            check(new Blob(["avif"], {type: "image/avif"})),
        ]).then((found) => console.log(JSON.stringify(found)));
    """

    def test_what_staging_reads_is_sent_as_it_is(self):
        png, _ = run(self.SCENARIO)

        assert png == {"same": True, "type": "image/png", "drawn": []}

    def test_anything_else_is_redrawn_as_a_jpeg_no_larger_than_it_needs_to_be(self):
        _, avif = run(self.SCENARIO)

        assert avif["same"] is False
        assert avif["type"] == "image/jpeg"
        assert avif["drawn"] == [[2048, 1024]]


class TestAPanelThatGrowsStaysOnTheWindow:
    """Found driving the flyout in Chromium: a panel docked along the bottom is
    placed from its own height, and grew downwards off the window as messages
    arrived -- and as a tap opened a message's actions -- taking the composer
    and Send with it. It is placed again whenever its height changes."""

    def test_a_change_of_height_places_it_again_and_nothing_else_does(self):
        found = run("""
            const shell = Object.create(NS.Shell.prototype);
            shell.state = {panelOpen: true};
            shell.nodes = {panel: {offsetHeight: 300}};
            let placed = 0;
            shell.place = () => { placed += 1; };
            shell.placedHeight = 300;
            const same = shell.resized();
            shell.nodes.panel.offsetHeight = 420;
            const grew = shell.resized();
            shell.state.panelOpen = false;
            shell.nodes.panel.offsetHeight = 500;
            const closed = shell.resized();
            console.log(JSON.stringify({same, grew, closed, placed}));
        """)

        assert found == {"same": False, "grew": True, "closed": False, "placed": 1}

    def test_placing_it_records_the_height_it_was_placed_at(self):
        from test_assistant_js import PANEL

        found = run(PANEL + """
            const shell = panel({state: {panelOpen: true, conversationExpanded: true}});
            shell.placeNow();
            console.log(JSON.stringify(shell.placedHeight));
        """)

        assert found == 140

    def test_the_panel_s_own_size_is_what_is_watched(self):
        shell = SHELL.read_text(encoding="utf-8")
        wire = shell.split("Shell.prototype.wire = function", 1)[1] \
            .split("Shell.prototype.grow", 1)[0]

        assert "new ResizeObserver(() => this.resized())" in wire
        assert "watched.observe(nodes.panel);" in wire
        assert "watched.disconnect()" in wire
