"""The Forge Assistant's panel hands its replies' speech to Voice Chat.

The user's report (2026-11-09): "Sometimes the voice chat TTS stops working
and does not recover." Their model_chain.log showed days of replies for which
the server made a speech turn, synthesised its first sentences and dropped it
thirty seconds later because no page ever opened its audio -- every one of them
a reply asked for from this panel, beside an image tab. The tab hands its turns
to Voice Chat through a hidden field; nothing handed the panel's over. The
command's answer now carries the turn (`speech_turn_id`) and the store passes
it to Voice Chat's facade (`followTurn`), and the panel's Stop stays up for as
long as the reply is read aloud, which goes on well after its last word.

Run against the real store and shell in the node harness `test_assistant_js.py`
uses, with a stand-in for Voice Chat's facade.
"""

from __future__ import annotations

from test_assistant_js import run

SPEAKER = """
// Voice Chat's facade as the store meets it: what it was handed, and the one
// listener the store subscribes.
const speaker = {
    followed: [], listeners: [], unsubscribed: 0, refuse: false,
    followTurn(turn) {
        if (speaker.refuse) throw new Error("the speaker is gone");
        speaker.followed.push(turn);
        return true;
    },
    subscribeSpeech(listener) {
        speaker.listeners.push(listener);
        listener({playing: false});
        return () => { speaker.unsubscribed += 1; };
    },
};
NS.speech = speaker;

// A store whose server answers each command as the scenario says.
function storeAnswering(answer) {
    const store = new NS.Store();
    store.readAloud = true;
    store.ensureFeed = () => Promise.resolve();
    store.review = () => undefined;
    store.after = () => undefined;
    store.request = () => Promise.resolve(answer);
    return store;
}
"""


def scenario(body):
    return run(SPEAKER + body, sources=("store",))


class TestTheTurnIsHandedOver:
    def test_an_accepted_reply_hands_its_turn_to_the_speaker(self):
        found = scenario("""
            const store = storeAnswering({ok: true, operation_id: "op1",
                                          speech_turn_id: "T1"});
            store.send({action: "send", payload: {text: "hi"}}).then((outcome) => {
                console.log(JSON.stringify({followed: speaker.followed,
                                            answered: outcome.speech_turn_id}));
            });
        """)

        assert found == {"followed": ["T1"], "answered": "T1"}

    def test_every_command_that_asks_for_a_reply_hands_it_over(self):
        found = scenario("""
            const store = storeAnswering({ok: true, speech_turn_id: "T1"});
            const actions = ["send", "regenerate", "continue", "resend_from_user"];
            Promise.all(actions.map((action) => store.send({action, payload: {}})))
                .then(() => console.log(JSON.stringify({count: speaker.followed.length})));
        """)

        assert found["count"] == 4

    def test_a_reply_that_is_not_to_be_spoken_hands_nothing_over(self):
        found = scenario("""
            const quiet = storeAnswering({ok: true, operation_id: "op1",
                                          speech_turn_id: ""});
            const refused = storeAnswering({ok: false, speech_turn_id: "T9",
                                            error: {code: "THREAD_BUSY"}});
            Promise.all([quiet.send({action: "send", payload: {}}),
                         refused.send({action: "send", payload: {}})]).then(() => {
                console.log(JSON.stringify({followed: speaker.followed}));
            });
        """)

        assert found["followed"] == []

    def test_a_command_that_asks_for_no_reply_hands_nothing_over(self):
        found = scenario("""
            const store = storeAnswering({ok: true, speech_turn_id: "T1"});
            store.send({action: "delete_message", payload: {}}).then(() => {
                console.log(JSON.stringify({followed: speaker.followed}));
            });
        """)

        assert found["followed"] == []

    def test_the_answer_to_a_command_sent_again_is_handed_over_too(self):
        """A send whose answer was lost is sent again with the same id, and the
        server answers it as a duplicate naming the same turn. That answer is
        the first this page hears of the turn; Voice Chat follows a turn once
        however often it is handed over."""
        found = scenario("""
            const store = storeAnswering({ok: true, duplicate: true,
                                          speech_turn_id: "T1"});
            store.send({action: "send", payload: {}}).then(() => {
                console.log(JSON.stringify({followed: speaker.followed}));
            });
        """)

        assert found["followed"] == ["T1"]

    def test_a_page_without_voice_chat_sends_as_it_did(self):
        found = scenario("""
            NS.speech = undefined;
            const store = storeAnswering({ok: true, speech_turn_id: "T1"});
            store.send({action: "send", payload: {}}).then((outcome) => {
                console.log(JSON.stringify({ok: outcome.ok, speaking: store.snapshot().speaking}));
            });
        """)

        assert found == {"ok": True, "speaking": False}

    def test_a_speaker_that_fails_does_not_fail_the_send(self):
        found = scenario("""
            speaker.refuse = true;
            const store = storeAnswering({ok: true, operation_id: "op1",
                                          speech_turn_id: "T1"});
            store.send({action: "send", payload: {}}).then((outcome) => {
                console.log(JSON.stringify({ok: outcome.ok, id: outcome.operation_id}));
            }, (error) => console.log(JSON.stringify({failed: String(error)})));
        """)

        assert found == {"ok": True, "id": "op1"}


class TestThePanelHearsTheSpeaker:
    def test_it_listens_once_and_says_when_a_reply_is_read_aloud(self):
        found = scenario("""
            const store = storeAnswering({ok: true, speech_turn_id: "T1"});
            let told = 0;
            store.subscribeState(() => { told += 1; });
            const seen = {};
            store.send({action: "send", payload: {}})
                .then(() => store.send({action: "regenerate", payload: {}}))
                .then(() => {
                    seen.listeners = speaker.listeners.length;
                    seen.before = store.snapshot().speaking;
                    const toldBefore = told;
                    speaker.listeners[0]({playing: true});
                    seen.reading = store.snapshot().speaking;
                    seen.toldOnStart = told > toldBefore;
                    const toldReading = told;
                    speaker.listeners[0]({playing: true});
                    seen.toldAgain = told > toldReading;
                    speaker.listeners[0]({playing: false});
                    seen.after = store.snapshot().speaking;
                    console.log(JSON.stringify(seen));
                });
        """)

        assert found == {"listeners": 1, "before": False, "reading": True,
                         "toldOnStart": True, "toldAgain": False, "after": False}

    def test_letting_the_store_go_lets_the_speaker_go(self):
        found = scenario("""
            const store = storeAnswering({ok: true, speech_turn_id: "T1"});
            store.send({action: "send", payload: {}}).then(() => {
                store.dispose();
                console.log(JSON.stringify({unsubscribed: speaker.unsubscribed}));
            });
        """)

        assert found["unsubscribed"] == 1


STOP = """
// The shell's render with every part it draws but the Stop stood in for.
const shell = Object.create(NS.Shell.prototype);
const made = (tag) => document.createElement(tag);
shell.nodes = {root: made("div"), unread: made("span"), input: made("textarea"),
               send: made("button"), stop: made("button")};
["grow", "renderChip", "renderAutoAttach", "renderReadAloud", "renderEmotion",
 "renderStart", "renderTranscript", "renderStatus", "applySuppression"]
    .forEach((name) => { shell[name] = () => undefined; });
shell.canSend = () => true;
function stopHidden(view) {
    shell.render(Object.assign({unreadTotal: 0, draft: {text: ""}, operation: null,
                                speech: {playing: false}, speaking: false}, view));
    return shell.nodes.stop.hidden;
}
"""


class TestThePanelsStop:
    def test_it_stays_up_while_the_reply_is_read_aloud(self):
        """The reply's text ends a minute or two before its reading does, and
        Stop was the operation's alone: it went with the last word."""
        found = run(STOP + """
            console.log(JSON.stringify({
                idle: stopHidden({}),
                reading: stopHidden({speaking: true}),
                writing: stopHidden({operation: {id: "op1", terminal: false}}),
                done: stopHidden({operation: {id: "op1", terminal: true}}),
            }));
        """, sources=("shell",))

        assert found == {"idle": True, "reading": False, "writing": False, "done": True}

    def test_pressing_it_while_reading_stops_the_page_s_speaker(self):
        found = run("""
            const shell = Object.create(NS.Shell.prototype);
            const calls = [];
            shell.store = {snapshot: () => ({operation: null}),
                           stop: (id) => calls.push("reply:" + id)};
            NS.speech = {stopPlayback: (options) => calls.push("speaker:" + options.origin)};
            shell.stopReply();
            console.log(JSON.stringify({calls}));
        """, sources=("shell",))

        assert found["calls"] == ["speaker:any"]
