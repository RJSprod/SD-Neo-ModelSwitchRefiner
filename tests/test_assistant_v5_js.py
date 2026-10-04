"""The flyout's fifth round: the header as glyphs, and the ⋯ menu in groups.

Asked for together:

    1. the line naming the character and the conversation goes, and so does
       the Conversation accordion -- a Chat button in the top row shows and
       hides the conversation instead;
    2. the Focus button goes; three presses on the drag area toggle focus
       (`test_assistant_taps_js.py`);
    3. the top row and the bottom row are shorter, and every button in them
       is a glyph;
    4. editing a message is a dialog (`test_assistant_v4_js.py`,
       `test_message_editor_js.py`);
    5. the ⋯ menu has New chat, and is grouped and nested.

Run against the real browser files in the node harness `test_assistant_js.py`
uses. Every test here was checked against the change it guards by reverting
that change and watching the test fail.
"""

from __future__ import annotations

import re

from test_assistant_js import SHELL, run

CSS = (SHELL.parent.parent / "style.css").read_text(encoding="utf-8")


def rule(selector):
    bare = re.sub(r"/\*.*?\*/", "", CSS, flags=re.DOTALL)
    found = []
    for match in re.finditer(r"([^{}]+)\{([^{}]*)\}", bare):
        names = [piece.strip() for piece in match.group(1).split(",")]
        if selector in names:
            found.append(match.group(2))
    assert found, selector + " has no rule of its own"
    return "\n".join(found)


# --------------------------------------------------------------------------- #
# 1. The Chat button
# --------------------------------------------------------------------------- #

CHAT = """
const made = (tag) => document.createElement(tag);
const shell = Object.create(NS.Shell.prototype);
shell.state = {conversationExpanded: true, panelOpen: true};
shell.saved = 0;
shell.placed = 0;
shell._save = () => { shell.saved += 1; };
shell.place = () => { shell.placed += 1; };
shell.rendered = 0;
shell.renderWorkspaces = () => { shell.rendered += 1; };
shell.closed = [];
shell.closeMenu = () => { shell.closed.push(shell.nodes.menu.dataset.owner); };
const menu = made("div");
menu.hidden = true;
menu.dataset.owner = "";
shell.nodes = {chat: made("button"), body: made("div"), panel: made("section"),
               workspaces: made("div"), picker: made("button"), menu};
const seen = () => ({
    expanded: shell.state.conversationExpanded,
    says: shell.nodes.chat["aria-expanded"],
    body: shell.nodes.body.hidden,
    collapsed: shell.nodes.panel.classList.contains("forge-assistant-collapsed"),
    strip: shell.nodes.workspaces.hidden,
    picker: shell.nodes.picker.hidden,
});
"""


class TestTheChatButton:
    """"Instead, i want a simple button in the top row that reads Chat.
    Tapping the chat button should have the same function as the expand and
    collapse today.\""""

    def test_it_shows_and_hides_the_whole_conversation(self):
        found = run(CHAT + """
            shell.applyChat();
            const open = seen();
            shell.toggleChat();
            const closed = seen();
            shell.toggleChat();
            console.log(JSON.stringify({open, closed, again: seen(), saved: shell.saved,
                                        placed: shell.placed}));
        """)

        assert found["open"] == {"expanded": True, "says": "true", "body": False,
                                 "collapsed": False, "strip": True, "picker": False}
        assert found["closed"] == {"expanded": False, "says": "false", "body": True,
                                   "collapsed": True, "strip": False, "picker": True}
        assert found["again"] == found["open"]
        assert found["saved"] == 2 and found["placed"] == 3, (
            "remembered each time, and the panel placed again for its new size")

    def test_collapsing_draws_the_workspace_row_and_closes_its_menu(self):
        found = run(CHAT + """
            shell.nodes.menu.dataset.owner = "workspaces";
            shell.state.conversationExpanded = false;
            shell.applyChat();
            shell.nodes.menu.dataset.owner = "utilities";
            shell.applyChat();
            console.log(JSON.stringify({rendered: shell.rendered, closed: shell.closed}));
        """)

        assert found == {"rendered": 2, "closed": ["workspaces"]}, (
            "only the menu whose button is about to be hidden")

    def test_the_button_is_a_glyph_with_its_word_for_a_screen_reader(self):
        found = run("""
            const shell = Object.create(NS.Shell.prototype);
            shell.settings = {label: "Forge Assistant", bubbleWidth: 80};
            shell.state = {conversationExpanded: true};
            shell.nodes = {root: {appendChild() {}}};
            shell.disposers = [];
            shell.host = {registerMenu: () => () => undefined,
                          getActiveWorkspace: () => "tab_txt2img",
                          listWorkspaces: () => []};
            shell.place = () => undefined;
            shell.buildPanel();
            const n = shell.nodes;
            const glyph = (node) => ({text: node.textContent, label: node["aria-label"],
                                      title: node.title, type: node.type});
            console.log(JSON.stringify({chat: glyph(n.chat), picker: glyph(n.picker),
                                        send: glyph(n.send), stop: glyph(n.stop),
                                        more: glyph(n.utilities),
                                        chatClass: n.chat.className,
                                        expanded: n.chat["aria-expanded"]}));
        """)

        for name, label in (("chat", "Chat"), ("picker", "Workspace"), ("send", "Send"),
                            ("stop", "Stop the reply"), ("more", "More actions")):
            glyph = found[name]
            assert glyph["label"] == label, name
            assert glyph["title"] == label, name
            assert glyph["type"] == "button", name
            assert len(glyph["text"]) <= 2 and glyph["text"] != label, (name, glyph["text"])
        assert "forge-assistant-chat" in found["chatClass"]
        assert found["expanded"] == "true"


class TestTheRowsAreShorter:
    """"I want the buttons rows to have reduced height ... convert all
    buttons with text to icons.\""""

    def test_the_header_s_and_the_composer_s_buttons_are_36_not_44(self):
        for selector in (".forge-assistant-nav-button", ".forge-assistant-icon-button",
                         ".forge-assistant-send", ".forge-assistant-stop"):
            assert "min-height: 36px" in rule(selector), selector

    def test_the_input_matches_them(self):
        assert "min-height: 36px" in rule(".forge-assistant-input")

    def test_a_list_s_entries_keep_the_full_finger(self):
        """The menus and the actions under a message are lists a thumb
        picks from; the rows are chrome."""
        assert "min-height: 44px" in rule(".forge-assistant-menu-item")
        assert "min-height: 44px" in rule(".forge-assistant-action")

    def test_chat_is_lit_while_the_conversation_shows(self):
        assert "var(--color-accent" in rule('.forge-assistant-chat[aria-expanded="true"]')

    def test_send_is_the_theme_s_primary_button(self):
        send = rule(".forge-assistant-send")

        assert "--button-primary-background-fill" in send
        assert "accent-soft" not in send

    def test_the_header_is_tighter(self):
        header = rule(".forge-assistant-header")

        assert "padding: 0.25em 0.4em" in header


# --------------------------------------------------------------------------- #
# 2. The ⋯ menu: groups, New chat, and a Threads submenu
# --------------------------------------------------------------------------- #

MENU = """
const made = (tag) => document.createElement(tag);
const shell = Object.create(NS.Shell.prototype);
shell.state = {freeFloat: false, autoAttach: false, sendToGenerate: false,
               conversationExpanded: true, panelOpen: true};
shell.host = {listUtilities: () => [], openOnly() { shell.onlyOpened = true; }};
let view = {selection: {character: "Ada", thread: "t1"},
            conversation: {threads: [{thread_id: "t1", title: "Harbour"},
                                     {thread_id: "t2", title: "Dusk"}]}};
shell.store = {snapshot: () => view, selected: [],
               select(c, t) { shell.store.selected.push([c, t]); }};
const menu = made("div");
menu.hidden = true;
Object.defineProperty(menu, "innerHTML", {get: () => "", set: () => { menu.children = []; }});
shell.nodes = {menu, picker: made("button"), utilities: made("button"),
               header: made("header"), status: made("p")};
shell.placed = 0;
shell.place = () => { shell.placed += 1; };
shell.menuHandle = {};
const labels = () => menu.children.map((c) => c.textContent);
const entries = () => menu.children.filter((c) => c.tagName === "BUTTON");
const press = (label) => {
    const item = menu.children.find((c) => c.textContent === label);
    item.handlers.click.forEach((fn) => fn());
    return item;
};
"""


class TestTheMenuIsGrouped:
    """"Reorganize the '...' menu, maybe add some nesting. Please recommend
    and implement enhanced menu.\""""

    def test_the_entries_are_in_four_groups_headed_by_what_they_act_on(self):
        found = run(MENU + """
            shell.host.listUtilities = () => [
                {id: "a", label: "Unload All Models", enabled: true, kind: "unload", scope: "all"},
                {id: "l", label: "Unload LLM", enabled: true, kind: "unload", scope: "llm"}];
            const items = shell.utilityItems();
            console.log(JSON.stringify(items.map((i) => [i.tagName, i.getAttribute("role"),
                                                         i.textContent])));
        """, sources=("shell", "system", "look"))

        assert found == [
            ["DIV", "presentation", "Chat · Ada"],
            ["BUTTON", "menuitem", "New chat"],
            ["BUTTON", "menuitem", "Threads"],
            ["BUTTON", "menuitem", "System prompt…"],
            ["DIV", "presentation", "Composer"],
            ["BUTTON", "menuitemcheckbox", "Auto Attach"],
            ["BUTTON", "menuitemcheckbox", "Send to Generate"],
            ["DIV", "presentation", "Panel"],
            ["BUTTON", "menuitemcheckbox", "Free Float"],
            ["BUTTON", "menuitem", "Customize…"],
            ["DIV", "presentation", "Models"],
            ["BUTTON", "menuitemradio", "Warm LoRA"],
            ["BUTTON", "menuitemradio", "Cold LoRA"],
            ["BUTTON", "menuitem", "Unload All Models"],
            ["BUTTON", "menuitem", "Unload LLM"],
        ]

    def test_the_chat_group_is_headed_by_the_character(self):
        """The line under the header that said who the panel is talking to is
        gone; this is where the name went."""
        found = run(MENU + """
            const withAda = shell.utilityItems()[0].textContent;
            view = {selection: {character: "", thread: ""}, conversation: null};
            const items = shell.utilityItems();
            console.log(JSON.stringify({withAda, without: items[0].textContent,
                newChat: [items[1].disabled, items[1].title],
                threads: [items[2].disabled, items[2].title]}));
        """)

        assert found["withAda"] == "Chat · Ada"
        assert found["without"] == "Chat"
        assert found["newChat"] == [True, "Choose a conversation first"]
        assert found["threads"] == [True, "Choose a conversation first"]

    def test_the_unload_entries_are_there_only_when_the_host_offers_something(self):
        """The Models group itself is always drawn since the Warm LoRA / Cold
        LoRA switch moved into it; what the host offers to unload still only
        appears when it offers it."""
        found = run(MENU + """
            console.log(JSON.stringify(shell.utilityItems().map((i) => i.textContent)));
        """)

        assert "Models" in found
        assert not [label for label in found if label.startswith("Unload")]

    def test_a_group_heading_is_not_an_entry(self):
        heading = rule(".forge-assistant-menu-group")

        assert "text-transform: uppercase" in heading
        source = SHELL.read_text(encoding="utf-8")
        group = source.split("Shell.prototype.groupLabel = function", 1)[1] \
            .split("Shell.prototype", 1)[0]
        assert '"presentation"' in group
        assert "addEventListener" not in group

    def test_new_chat_says_what_it_does(self):
        found = run(MENU + """
            const item = shell.utilityItems()[1];
            console.log(JSON.stringify({label: item.textContent, title: item.title,
                                        disabled: item.disabled}));
        """)

        assert found == {"label": "New chat", "title": "Start a new chat with Ada",
                         "disabled": False}


class TestTheThreadsSubmenu:
    def test_threads_opens_one_level_down_in_the_same_box(self):
        found = run(MENU + """
            shell.toggleMenu("utilities");
            const top = {which: menu.dataset.which, owner: menu.dataset.owner,
                         placed: shell.placed};
            const threads = press("Threads");
            console.log(JSON.stringify({top, popup: threads["aria-haspopup"],
                which: menu.dataset.which, owner: menu.dataset.owner, placed: shell.placed,
                labels: labels(), hidden: menu.hidden,
                roles: entries().map((e) => e.getAttribute("role")),
                current: entries().map((e) => e.getAttribute("aria-current"))}));
        """)

        assert found["top"] == {"which": "utilities", "owner": "utilities", "placed": 1}
        assert found["popup"] == "menu"
        assert found["which"] == "threads" and found["owner"] == "utilities"
        assert found["placed"] == 2, "the panel placed again for its new height"
        assert found["hidden"] is False
        assert found["labels"] == ["‹ Back", "Threads · Ada", "Harbour", "Dusk", "Cancel"]
        assert found["roles"] == ["menuitem", "menuitem", "menuitem", "menuitem"]
        assert found["current"] == [None, "true", "false", None]

    def test_choosing_a_thread_moves_the_panel_and_closes_the_menu(self):
        found = run(MENU + """
            shell.toggleMenu("utilities");
            press("Threads");
            press("Dusk");
            console.log(JSON.stringify({selected: shell.store.selected, hidden: menu.hidden,
                                        owner: menu.dataset.owner}));
        """)

        assert found == {"selected": [["Ada", "t2"]], "hidden": True, "owner": ""}

    def test_back_returns_to_the_menu_s_own_list(self):
        found = run(MENU + """
            shell.toggleMenu("utilities");
            press("Threads");
            press("\\u2039 Back");
            console.log(JSON.stringify({which: menu.dataset.which, owner: menu.dataset.owner,
                                        first: labels()[0], last: labels().slice(-1)[0],
                                        hidden: menu.hidden}));
        """)

        assert found == {"which": "utilities", "owner": "utilities", "first": "Chat · Ada",
                         "last": "Cancel", "hidden": False}

    def test_the_menu_s_own_button_still_closes_it_from_a_submenu(self):
        """The ⋯ opened it, so ⋯ closes it, whichever list is showing."""
        found = run(MENU + """
            shell.toggleMenu("utilities");
            press("Threads");
            shell.toggleMenu("utilities");
            console.log(JSON.stringify({hidden: menu.hidden, which: menu.dataset.which,
                                        expanded: shell.nodes.utilities["aria-expanded"]}));
        """)

        assert found == {"hidden": True, "which": "", "expanded": "false"}

    def test_with_no_threads_the_submenu_says_so_and_still_has_a_way_out(self):
        found = run(MENU + """
            view = {selection: {character: "Ada", thread: ""}, conversation: {threads: []}};
            shell.toggleMenu("utilities");
            const item = press("Threads");
            console.log(JSON.stringify({title: item.title, labels: labels()}));
        """)

        assert found["title"] == "No threads yet"
        assert found["labels"] == ["‹ Back", "Threads · Ada",
                                   "No threads yet. Start one with New chat.", "Cancel"]

    def test_the_chevron_is_the_stylesheet_s_so_the_word_stays_the_word(self):
        assert 'content: "\\203A"' in rule(".forge-assistant-submenu::after")


# --------------------------------------------------------------------------- #
# 3. The panel and the tab are one conversation
# --------------------------------------------------------------------------- #

STUDIO = """
// The Conversation tab's open bridge, as Gradio draws it: a box inside a
// holder carrying the id, and a button that is its own holder.
const pressed = [];
const events = [];
const box = {tagName: "TEXTAREA", value: "",
             dispatchEvent(event) { events.push(event.type + ":" + event.bubbles); }};
const holder = {id: "mc-llm-chat-open-at", tagName: "DIV",
                querySelector: (selector) => selector === "textarea" ? box : null};
const button = {id: "mc-llm-chat-open-now", tagName: "BUTTON",
                click() { pressed.push(box.value); }};
let page = {"mc-llm-chat-open-at": holder, "mc-llm-chat-open-now": button};
document.getElementById = (id) => page[id] || null;
const shell = Object.create(NS.Shell.prototype);
"""


class TestNewChatMovesTheTabToo:
    """"A new thread should make the flyout and conversation mode start from
    scratch because they are in sync." The panel used to move alone; the next
    look found the tab on the old thread and brought the panel back to it."""

    def test_the_tab_is_told_which_thread_through_its_own_controls(self):
        found = run(STUDIO + """
            const moved = shell.openInStudio("t9");
            console.log(JSON.stringify({moved, value: box.value, events, pressed}));
        """)

        assert found == {"moved": True, "value": "t9", "events": ["input:true"],
                         "pressed": ["t9"]}

    def test_a_page_without_the_tab_s_controls_moves_nothing(self):
        found = run(STUDIO + """
            page = {};
            const moved = shell.openInStudio("t9");
            const none = shell.openInStudio("");
            console.log(JSON.stringify({moved, none, pressed}));
        """)

        assert found == {"moved": False, "none": False, "pressed": []}

    def test_new_chat_opens_the_new_thread_in_the_tab(self):
        found = run(STUDIO + """
            shell.state = {conversationExpanded: true};
            shell.nodes = {status: document.createElement("p")};
            shell.store = {createThread: () => Promise.resolve({ok: true,
                resulting_conversation: {character: "Ada", thread_id: "t9"}})};
            shell.startThread("Ada").then(() => console.log(JSON.stringify(
                {pressed, said: shell.nodes.status.textContent})));
        """)

        assert found == {"pressed": ["t9"], "said": "New chat with Ada."}

    def test_a_refused_new_chat_moves_the_tab_nowhere(self):
        found = run(STUDIO + """
            shell.state = {conversationExpanded: true};
            shell.nodes = {status: document.createElement("p")};
            shell.store = {createThread: () => Promise.resolve({ok: false, error: {message: "No."}})};
            shell.startThread("Ada").then(() => console.log(JSON.stringify({pressed})));
        """)

        assert found == {"pressed": []}

    def test_choosing_a_thread_in_the_panel_opens_it_in_the_tab(self):
        found = run(STUDIO + """
            shell.closeMenu = () => {};
            const selected = [];
            shell.store = {snapshot: () => ({selection: {character: "Ada", thread: "t1"},
                conversation: {threads: [{thread_id: "t1", title: "Harbour"},
                                         {thread_id: "t2", title: "Dusk"}]}}),
                select(c, t) { selected.push([c, t]); }};
            const item = shell.threadItems().find((i) => i.textContent === "Dusk");
            item.handlers.click.forEach((fn) => fn());
            console.log(JSON.stringify({selected, pressed}));
        """)

        assert found == {"selected": [["Ada", "t2"]], "pressed": ["t2"]}


# --------------------------------------------------------------------------- #
# Warm LoRA, Cold LoRA: where the loaded LoRA's originals live
# --------------------------------------------------------------------------- #


class TestWarmAndColdLora:
    """"I want there to be a toggle in the flyout menus '...' menu to turn this
    setting on and off. 'Warm Lora' or 'Cold Lora'.\""""

    def test_the_two_entries_are_one_radio_pair_with_the_current_mode_checked(self):
        found = run(MENU + """
            const checked = () => shell.utilityItems()
                .filter((i) => i.getAttribute("role") === "menuitemradio")
                .map((i) => [i.textContent, i.getAttribute("aria-checked")]);
            const unknown = checked();
            shell.loraRam = {mode: "cold", merged: false, blob: true, originals_bytes: 0};
            const cold = checked();
            shell.loraRam = {mode: "warm", merged: true, blob: false, originals_bytes: 12.1 * 2 ** 30};
            const warm = checked();
            const titles = shell.utilityItems()
                .filter((i) => i.getAttribute("role") === "menuitemradio").map((i) => i.title);
            console.log(JSON.stringify({unknown, cold, warm, titles}));
        """, sources=("shell", "system", "look"))

        # Before the server has said anything the entries read as Forge's way.
        assert found["unknown"] == [["Warm LoRA", "true"], ["Cold LoRA", "false"]]
        assert found["cold"] == [["Warm LoRA", "false"], ["Cold LoRA", "true"]]
        assert found["warm"] == [["Warm LoRA", "true"], ["Cold LoRA", "false"]]
        assert "12.1 GB of un-merged weights in system RAM now" in found["titles"][0]
        assert "reloads the checkpoint from disk" in found["titles"][1]

    def test_the_models_group_is_there_with_nothing_to_unload(self):
        found = run(MENU + """
            console.log(JSON.stringify(shell.utilityItems().map((i) => i.textContent)));
        """, sources=("shell", "system", "look"))

        assert found[-3:] == ["Models", "Warm LoRA", "Cold LoRA"]

    def test_pressing_the_other_one_posts_the_mode_closes_the_menu_and_tells_the_answer(self):
        found = run(MENU + """
            shell.loraRam = {mode: "warm", merged: true, blob: false, originals_bytes: 12.1 * 2 ** 30};
            shell.store.request = (path, options) => {
                shell.asked = {path, method: options && options.method};
                return Promise.resolve({ok: true, lora: {mode: "cold", merged: false, blob: true},
                                        message: "Cold LoRA: 12.1 GB of un-merged weights freed from system RAM."});
            };
            shell.closed = 0;
            shell.closeMenu = () => { shell.closed += 1; };
            shell.showMenu("utilities");
            const item = menu.children.find((c) => c.textContent === "Cold LoRA");
            item.handlers.click.forEach((fn) => fn());
            setImmediate(() => console.log(JSON.stringify({
                asked: shell.asked, closed: shell.closed, mode: shell.loraRam.mode,
                told: shell.told && shell.told.text,
                checked: shell.utilityItems().filter((i) => i.getAttribute("role") === "menuitemradio")
                    .map((i) => i.getAttribute("aria-checked"))})));
        """, sources=("shell", "system", "look"))

        assert found["asked"] == {"path": "/lora-ram?mode=cold", "method": "POST"}
        assert found["closed"] == 1
        assert found["mode"] == "cold"
        assert found["told"].startswith("Cold LoRA: 12.1 GB")
        # The next menu reads the server's answer, not the press.
        assert found["checked"] == ["false", "true"]

    def test_pressing_the_one_already_on_asks_nothing(self):
        found = run(MENU + """
            shell.loraRam = {mode: "warm"};
            shell.asked = [];
            shell.store.request = (path) => { shell.asked.push(path); return Promise.resolve({}); };
            shell.closeMenu = () => {};
            shell.showMenu("utilities");
            const item = menu.children.find((c) => c.textContent === "Warm LoRA");
            item.handlers.click.forEach((fn) => fn());
            console.log(JSON.stringify(shell.asked));
        """, sources=("shell", "system", "look"))

        assert found == []

    def test_a_refused_change_is_said_and_the_mode_stays(self):
        found = run(MENU + """
            shell.loraRam = {mode: "warm"};
            shell.store.request = () => Promise.reject(new Error("Reload the page"));
            shell.closeMenu = () => {};
            shell.setLoraRam("cold").then(() => console.log(JSON.stringify({
                mode: shell.loraRam.mode, told: shell.told && [shell.told.text, shell.told.kind]})));
        """, sources=("shell", "system", "look"))

        assert found == {"mode": "warm", "told": ["Reload the page", "warn"]}

