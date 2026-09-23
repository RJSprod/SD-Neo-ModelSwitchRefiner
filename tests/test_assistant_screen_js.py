"""Focus mode takes the browser's full screen with it, and gives it back.

Asked for: "One press of the toggle = tabbar hidden & browser goes full screen
view... press toggle again = tabbar returns & exit full page browser view."

So the Focus toggle asks the browser for full screen from inside the press
that turns focus on -- the only moment a browser will grant it -- and ends it
when focus goes off. The rest is what keeps the two in step:

    the whole document goes full screen, never the workspace, because an
    element in full screen is the only thing drawn and the way out of focus
    is the assistant, which is outside it;

    the browser ending full screen by its own means (Escape, a back gesture,
    a switch of tab) turns focus off, so the toggle never disagrees with the
    screen;

    only a full screen the assistant asked for is ever ended by it -- a video
    somebody is watching, or anything that was full screen before, is left;

    a switch of workspace inside focus keeps the full screen, although focus
    itself comes off and goes back on for it;

    and every answer the browser can give -- granted, refused, never, too
    late -- leaves the toggle telling the truth.

Every test here was checked against the change it guards by reverting that
change and watching the test fail.
"""

from __future__ import annotations

import re

from test_assistant_js import SHELL, run


# A browser that can go full screen and answers when the test says so, and a
# shell with just enough around it to press Focus. The focus registry is a
# stub: the transaction itself is test_assistant_focus_js's business, and here
# it only has to be on or off.
SCREEN = """
const page = document.documentElement;
const video = {tagName: "VIDEO"};

function browser(options) {
    options = options || {};
    const calls = [];
    let answer = null;
    document.fullscreenEnabled = options.enabled !== false;
    document.fullscreenElement = options.already || null;
    if (options.api !== false) {
        page.requestFullscreen = function (settings) {
            calls.push({request: this === page ? "page" : "other",
                        navigationUI: (settings && settings.navigationUI) || null});
            if (options.throws) throw new TypeError("requestFullscreen is not allowed");
            return new Promise((resolve, reject) => { answer = {resolve, reject}; });
        };
        document.exitFullscreen = () => {
            calls.push("exit");
            document.fullscreenElement = null;
            return Promise.resolve();
        };
    }
    return {
        calls,
        requests: () => calls.filter((call) => call.request).length,
        exits: () => calls.filter((call) => call === "exit").length,
        grant() {
            document.fullscreenElement = page;
            const pending = answer;
            answer = null;
            pending.resolve();
        },
        refuse() {
            const pending = answer;
            answer = null;
            pending.reject(new TypeError("Permissions check failed"));
        },
        // A promise that settles after the page has already left again.
        grantAndLose() {
            document.fullscreenElement = null;
            const pending = answer;
            answer = null;
            pending.resolve();
        },
    };
}

// Whatever the browser does is announced with an event; the harness has no
// dispatch, so the test announces it by calling what the event is wired to.
const settle = () => new Promise((resolve) => setImmediate(resolve));

function focusShell(over) {
    over = over || {};
    const shell = Object.create(NS.Shell.prototype);
    shell.state = {panelOpen: true, focusEnabled: false, focusWorkspaceId: null};
    shell.saved = 0;
    shell.said = [];
    shell._save = () => { shell.saved += 1; };
    shell.place = () => {};
    shell.closeMenu = () => {};
    shell.say = (text, tone) => shell.said.push({text, tone});
    shell.nodes = {focusToggle: {setAttribute(name, value) { this[name] = value; }}};
    shell.host = Object.assign({
        getActiveWorkspace: () => "tab_txt2img",
        activateWorkspace: (id) => Promise.resolve(id),
    }, over.host || {});
    shell.focus = Object.assign({
        on: "",
        refuse: new Set(),
        isActive() { return !!this.on; },
        activeWorkspace() { return this.on; },
        exit() { const was = !!this.on; this.on = ""; return was; },
        enter(id) {
            if (this.refuse.has(id)) return {ok: false, reason: "Not here.", note: ""};
            this.on = id;
            return {ok: true, reason: "", note: ""};
        },
    }, over.focus || {});
    return shell;
}

function seen(shell, screen) {
    return {
        focus: shell.focus.on,
        enabled: shell.state.focusEnabled,
        pressed: shell.nodes.focusToggle["aria-pressed"],
        owned: !!shell.screenOwned,
        pending: !!shell.screenPending,
        full: document.fullscreenElement === page ? "page"
            : (document.fullscreenElement ? "other" : null),
        requests: screen.requests(),
        exits: screen.exits(),
        calls: screen.calls,
    };
}
"""


class TestOnePressBothWays:
    def test_the_press_that_turns_focus_on_asks_for_the_whole_page(self):
        """The document, never the workspace: an element in full screen is
        the only thing the browser draws, and the assistant -- the way back
        out -- is not inside the workspace."""
        found = run(SCREEN + """
            const screen = browser();
            const shell = focusShell();
            shell.toggleFocus();
            const asked = seen(shell, screen);
            screen.grant();
            await settle();
            shell.screenChanged();
            console.log(JSON.stringify({asked, granted: seen(shell, screen)}));
        """)

        assert found["asked"]["focus"] == "tab_txt2img"
        assert found["asked"]["pressed"] == "true"
        assert found["asked"]["calls"] == [{"request": "page", "navigationUI": "hide"}]
        assert found["granted"]["owned"] is True
        assert found["granted"]["full"] == "page"
        assert found["granted"]["pending"] is False

    def test_the_second_press_ends_both(self):
        found = run(SCREEN + """
            const screen = browser();
            const shell = focusShell();
            shell.toggleFocus();
            screen.grant();
            await settle();
            shell.toggleFocus();
            console.log(JSON.stringify(seen(shell, screen)));
        """)

        assert found["focus"] == ""
        assert found["pressed"] == "false"
        assert found["exits"] == 1
        assert found["full"] is None
        assert found["owned"] is False

    def test_off_and_on_again_asks_again(self):
        """Ownership ends with the full screen: a flag left set would make
        the next press think the page was still full screen and ask for
        nothing."""
        found = run(SCREEN + """
            const screen = browser();
            const shell = focusShell();
            shell.toggleFocus();
            screen.grant();
            await settle();
            shell.toggleFocus();
            shell.screenChanged();          // the exit's own change event
            shell.toggleFocus();
            console.log(JSON.stringify(seen(shell, screen)));
        """)

        assert found["requests"] == 2
        assert found["focus"] == "tab_txt2img"

    def test_escape_on_the_page_ends_both(self):
        """Escape leaves focus through the toggle. In full screen the browser
        usually keeps that key for itself (see the next class); where the page
        does get it, it must not leave the browser full screen over a tab bar
        that has come back."""
        found = run(SCREEN + """
            const screen = browser();
            const shell = focusShell();
            shell.toggleFocus();
            screen.grant();
            await settle();
            NS.escapeOrder = () => "exit-focus";
            NS.hostInDialog = () => false;
            shell.documentKey({key: "Escape", preventDefault() {}, stopPropagation() {}});
            console.log(JSON.stringify(seen(shell, screen)));
        """)

        assert found["focus"] == ""
        assert found["exits"] == 1

    def test_focus_that_cannot_be_entered_asks_for_nothing(self):
        found = run(SCREEN + """
            const screen = browser();
            const shell = focusShell();
            shell.focus.refuse.add("tab_txt2img");
            shell.toggleFocus();
            console.log(JSON.stringify(seen(shell, screen)));
        """)

        assert found["focus"] == ""
        assert found["requests"] == 0


class TestTheBrowserEndingItEndsFocus:
    def test_escape_or_a_back_gesture_turns_focus_off(self):
        """The browser ends full screen without asking anybody. Focus follows
        it off, so the toggle and the screen agree -- and nothing asks the
        browser to leave a full screen it has already left."""
        found = run(SCREEN + """
            const screen = browser();
            const shell = focusShell();
            shell.toggleFocus();
            screen.grant();
            await settle();
            const saved = shell.saved;
            document.fullscreenElement = null;          // the browser's Escape
            shell.screenChanged();
            console.log(JSON.stringify(Object.assign(seen(shell, screen),
                                                     {savedAgain: shell.saved > saved})));
        """)

        assert found["focus"] == ""
        assert found["enabled"] is False
        assert found["pressed"] == "false"
        assert found["exits"] == 0
        assert found["owned"] is False
        assert found["savedAgain"] is True

    def test_a_video_stacked_on_top_is_not_the_page_leaving(self):
        """A video made full screen inside focus changes the full-screen
        element without ending ours."""
        found = run(SCREEN + """
            const screen = browser();
            const shell = focusShell();
            shell.toggleFocus();
            screen.grant();
            await settle();
            document.fullscreenElement = video;
            shell.screenChanged();
            const stacked = seen(shell, screen);
            document.fullscreenElement = page;          // the video came back out
            shell.screenChanged();
            console.log(JSON.stringify({stacked, back: seen(shell, screen)}));
        """)

        assert found["stacked"]["focus"] == "tab_txt2img"
        assert found["stacked"]["owned"] is True
        assert found["back"]["focus"] == "tab_txt2img"
        assert found["back"]["owned"] is True


class TestOnlyItsOwnFullScreen:
    def test_something_already_full_screen_is_neither_claimed_nor_ended(self):
        """A full screen that was there before focus -- the page made full
        screen by something else, or a video -- is not the assistant's to
        end when focus goes off."""
        found = run(SCREEN + """
            const out = {};
            for (const [name, already] of [["page", page], ["video", video]]) {
                const screen = browser({already});
                const shell = focusShell();
                shell.toggleFocus();
                const on = seen(shell, screen);
                shell.toggleFocus();
                out[name] = {on, off: seen(shell, screen)};
            }
            console.log(JSON.stringify(out));
        """)

        for name in ("page", "video"):
            assert found[name]["on"]["focus"] == "tab_txt2img", name
            assert found[name]["on"]["requests"] == 0, name
            assert found[name]["off"]["focus"] == "", name
            assert found[name]["off"]["exits"] == 0, name
            assert found[name]["off"]["full"] is not None, name

    def test_focus_going_off_under_a_video_does_not_end_the_video(self):
        """Focus can go off while a video is full screen on top of the page
        -- a workspace that could not be refocused, say. Ending the full
        screen then would end the video being watched."""
        found = run(SCREEN + """
            const screen = browser();
            const shell = focusShell();
            shell.toggleFocus();
            screen.grant();
            await settle();
            document.fullscreenElement = video;
            shell.screenChanged();
            shell.focus.exit();
            shell.focusOff();
            console.log(JSON.stringify(seen(shell, screen)));
        """)

        assert found["enabled"] is False
        assert found["exits"] == 0
        assert found["full"] == "other"
        assert found["owned"] is False


class TestAWorkspaceSwitchKeepsIt:
    def test_switching_inside_focus_does_not_leave_full_screen(self):
        """Focus comes off for a switch and goes back on at the destination
        (see `switchWorkspace`). The full screen must not go with it: ending
        and re-entering it is a flash of the browser's bars on every switch,
        and the second request is made outside the press, where a browser may
        refuse it."""
        found = run(SCREEN + """
            const screen = browser();
            const shell = focusShell();
            shell.toggleFocus();
            screen.grant();
            await settle();
            await shell.switchWorkspace("tab_img2img");
            console.log(JSON.stringify(seen(shell, screen)));
        """)

        assert found["focus"] == "tab_img2img"
        assert found["requests"] == 1
        assert found["exits"] == 0
        assert found["owned"] is True

    def test_a_switch_that_lands_nowhere_ends_it_with_focus(self):
        found = run(SCREEN + """
            const screen = browser();
            const shell = focusShell({host: {
                activateWorkspace: () => Promise.reject(new Error("did not open")),
            }});
            shell.toggleFocus();
            screen.grant();
            await settle();
            shell.focus.refuse.add("tab_txt2img");      // not even back where it was
            await shell.switchWorkspace("tab_img2img");
            console.log(JSON.stringify(seen(shell, screen)));
        """)

        assert found["enabled"] is False
        assert found["exits"] == 1

    def test_every_way_focus_goes_off_goes_through_one_place(self):
        """The toggle, a refused refocus, a covered launcher, a workspace
        that could not be followed, the browser leaving full screen: one
        function turns focus off, so none of them can forget the screen."""
        shell = SHELL.read_text(encoding="utf-8")
        writes = re.findall(r"state\.focusEnabled = false", shell)
        body = shell.split("Shell.prototype.focusOff = function", 1)[1] \
            .split("Shell.prototype.", 1)[0]

        assert len(writes) == 1
        assert "state.focusEnabled = false" in body
        assert "this.leaveScreen()" in body


class TestEveryAnswerTheBrowserCanGive:
    def test_a_refusal_leaves_focus_as_it_always_was(self):
        """A frame without `allowfullscreen`, a press the browser did not
        count: focus stands on its own, and the next press may ask again."""
        found = run(SCREEN + """
            const screen = browser();
            const shell = focusShell();
            shell.toggleFocus();
            screen.refuse();
            await settle();
            const refused = seen(shell, screen);
            shell.toggleFocus();
            const off = seen(shell, screen);
            shell.toggleFocus();
            console.log(JSON.stringify({refused, off, again: seen(shell, screen)}));
        """)

        assert found["refused"]["focus"] == "tab_txt2img"
        assert found["refused"]["owned"] is False
        assert found["refused"]["pending"] is False
        assert found["off"]["exits"] == 0
        assert found["again"]["requests"] == 2

    def test_a_request_that_throws_is_a_refusal(self):
        found = run(SCREEN + """
            const screen = browser({throws: true});
            const shell = focusShell();
            shell.toggleFocus();
            console.log(JSON.stringify(seen(shell, screen)));
        """)

        assert found["focus"] == "tab_txt2img"
        assert found["pending"] is False
        assert found["owned"] is False

    def test_no_full_screen_at_all_is_focus_as_it_was(self):
        """iPhone Safari gives full screen to videos and nothing else."""
        found = run(SCREEN + """
            const out = {};
            for (const [name, options] of [["disabled", {enabled: false}],
                                           ["absent", {api: false}]]) {
                const screen = browser(options);
                delete page.requestFullscreen;
                if (name === "disabled") page.requestFullscreen = () => {
                    screen.calls.push({request: "page"});
                    return Promise.resolve();
                };
                const shell = focusShell();
                shell.toggleFocus();
                out[name] = seen(shell, screen);
            }
            console.log(JSON.stringify(out));
        """)

        for name in ("disabled", "absent"):
            assert found[name]["focus"] == "tab_txt2img", name
            assert found[name]["requests"] == 0, name
            assert found[name]["pending"] is False, name

    def test_focus_off_before_the_answer_gives_the_screen_back(self):
        """Two quick presses: the browser grants the first after the second
        turned focus off. Kept, it would be a full screen with the tab bar
        back and nothing on screen claiming it."""
        found = run(SCREEN + """
            const screen = browser();
            const shell = focusShell();
            shell.toggleFocus();
            shell.toggleFocus();
            screen.grant();
            await settle();
            shell.screenChanged();
            console.log(JSON.stringify(seen(shell, screen)));
        """)

        assert found["focus"] == ""
        assert found["exits"] == 1
        assert found["full"] is None
        assert found["owned"] is False

    def test_off_and_on_before_the_answer_asks_once_and_keeps_it(self):
        found = run(SCREEN + """
            const screen = browser();
            const shell = focusShell();
            shell.toggleFocus();
            shell.toggleFocus();
            shell.toggleFocus();
            screen.grant();
            await settle();
            console.log(JSON.stringify(seen(shell, screen)));
        """)

        assert found["requests"] == 1
        assert found["exits"] == 0
        assert found["focus"] == "tab_txt2img"
        assert found["owned"] is True

    def test_a_grant_already_over_does_not_leave_the_next_press_waiting(self):
        """Settled is settled: a request left marked pending would make every
        later press wait on an answer that has already come."""
        found = run(SCREEN + """
            const screen = browser();
            const shell = focusShell();
            shell.toggleFocus();
            screen.grantAndLose();
            await settle();
            shell.toggleFocus();
            shell.toggleFocus();
            console.log(JSON.stringify(seen(shell, screen)));
        """)

        assert found["requests"] == 2

    def test_an_answer_that_never_comes_is_not_waited_on_for_ever(self):
        found = run(SCREEN + """
            const screen = browser();
            const shell = focusShell();
            shell.toggleFocus();
            shell.toggleFocus();
            shell.screenPending = Date.now() - 60000;   // asked a minute ago
            shell.toggleFocus();
            console.log(JSON.stringify(seen(shell, screen)));
        """)

        assert found["requests"] == 2

    def test_another_elements_refusal_is_not_ours(self):
        """The error event is fired at the element that asked, and bubbles
        to the document -- a video's refusal must not cancel a request of
        ours that is still waiting."""
        found = run(SCREEN + """
            const screen = browser();
            const shell = focusShell();
            shell.toggleFocus();
            shell.screenRefused({target: video});
            const stillWaiting = !!shell.screenPending;
            shell.screenRefused({target: page});
            console.log(JSON.stringify({stillWaiting, after: !!shell.screenPending}));
        """)

        assert found["stillWaiting"] is True
        assert found["after"] is False

    def test_safari_s_prefixed_names_are_enough(self):
        """Safari before 16.4, and iPadOS with it, has only the webkit names,
        and its request answers with an event rather than a promise."""
        found = run(SCREEN + """
            const calls = [];
            delete page.requestFullscreen;
            delete document.exitFullscreen;
            document.fullscreenEnabled = undefined;
            document.fullscreenElement = undefined;
            document.webkitFullscreenEnabled = true;
            document.webkitFullscreenElement = null;
            page.webkitRequestFullscreen = function (flags) {
                calls.push({request: this === page ? "page" : "other", flags: flags === undefined
                            ? "none" : typeof flags});
            };
            document.webkitExitFullscreen = () => {
                calls.push("exit");
                document.webkitFullscreenElement = null;
            };
            const shell = focusShell();
            shell.toggleFocus();
            document.webkitFullscreenElement = page;
            shell.screenChanged();
            const owned = !!shell.screenOwned;
            shell.toggleFocus();
            console.log(JSON.stringify({calls, owned, focus: shell.focus.on}));
        """)

        assert found["calls"][0] == {"request": "page", "flags": "none"}
        assert found["owned"] is True
        assert found["calls"][-1] == "exit"
        assert found["focus"] == ""


class TestTheHooksAreWired:
    def test_the_change_and_error_events_under_both_names(self):
        shell = SHELL.read_text(encoding="utf-8")
        wiring = shell.split("Shell.prototype.wire = function", 1)[1] \
            .split("Shell.prototype.grow = function", 1)[0]

        assert re.search(r'\["fullscreenchange", "webkitfullscreenchange"\]\.forEach\(\(type\) => \{'
                         r'\s*this\.on\(document, type, \(\) => this\.screenChanged\(\)\);', wiring)
        assert re.search(r'\["fullscreenerror", "webkitfullscreenerror"\]\.forEach\(\(type\) => \{'
                         r'\s*this\.on\(document, type, \(event\) => this\.screenRefused\(event\)\);',
                         wiring)

