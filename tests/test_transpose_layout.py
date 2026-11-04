"""Transpose View, in Chromium: the tab bar and the panel's buttons as rails.

Asked for (docs/22 §3.47): "a new mode in flyout menu "..." menu that is
called "transpose view". it should me the tab bar of webui render on a slim
strip on the left of screen, all of its [icon] buttons, separated between tabs
and other stuff. Same on flyout menu buttons ... transpose those to a slim
strip on the left of the flyout, including the drag handle ... in both cases
the top of the view should be minimally padded content. also need an option to
choose right side for either of these independently", and then "i DO want this
functionality in focus mode as well".

The page is the theme's Forge-shaped page with the Forge Assistant's scripts
on it (`tests/test_theme_layout.py`): Forge's own tab bar under `#tabs`, the
theme's header drawn on it, fourteen tabs. What is measured is what the user
sees: where the rail is, where the page and the panel's column start, that
nothing sits under a rail, and that focus mode, the menu and the panel's
placement follow.
"""

from __future__ import annotations

import re

import pytest

from test_theme_layout import (ICONS, ON, USER_CSS_WHOLE, browser, enter_focus,  # noqa: F401
                               focus_page, open_page, settle)

RAIL = 48          # the tab rail's width (style.css, --mc-theme-rail)
PANEL_RAIL = 44    # the panel rail's width (style.css, --forge-assistant-rail-width)

GEOMETRY = """() => {
    const box = (node) => {
        if (!node) return null;
        const r = node.getBoundingClientRect();
        return {left: r.left, top: r.top, right: r.right, bottom: r.bottom, width: r.width, height: r.height};
    };
    const nav = document.querySelector('#tabs > .tab-nav');
    const tabs = [...nav.querySelectorAll(':scope > button[role="tab"]')];
    const actions = document.querySelector('.mc-theme-actions');
    const panel = document.querySelector('#tab_txt2img');
    const root = document.querySelector('.forge-assistant-focus-root');
    return {
        rail: document.documentElement.getAttribute('data-forge-assistant-rail'),
        icons: document.documentElement.getAttribute('data-mc-theme-tabs'),
        nav: box(nav), position: getComputedStyle(nav).position,
        tabs: tabs.map(box), glyphs: tabs.filter((t) => t.querySelector('svg')).length,
        actions: box(actions), actionsLine: getComputedStyle(actions).borderTopWidth,
        actionKids: [...actions.children].filter((c) => !c.hidden && getComputedStyle(c).display !== 'none').map(box),
        content: box(panel), root: box(root),
        focusRail: document.body.getAttribute('data-forge-assistant-focus-rail'),
        accent: getComputedStyle(nav.querySelector(':scope > button[role="tab"].selected')).boxShadow,
        ground: getComputedStyle(nav, '::before').content, strips: getComputedStyle(nav).boxShadow,
        width: document.documentElement.clientWidth, height: innerHeight,
    };
}"""

PANEL = """() => {
    const box = (node) => {
        if (!node) return null;
        const r = node.getBoundingClientRect();
        return {left: r.left, top: r.top, right: r.right, bottom: r.bottom, width: r.width, height: r.height};
    };
    const panel = document.querySelector('.forge-assistant-panel');
    const header = panel.querySelector('.forge-assistant-header');
    const shown = (node) => !node.hidden && getComputedStyle(node).display !== 'none';
    const buttons = [...header.children].filter((c) => c.tagName === 'BUTTON' && shown(c));
    const column = [...panel.children].filter((c) => c !== header && shown(c)
        && !c.classList.contains('forge-assistant-corner') && !c.classList.contains('forge-assistant-resize'));
    return {
        rail: panel.getAttribute('data-rail'), panel: box(panel), header: box(header),
        buttons: buttons.map(box), grip: box(panel.querySelector('.forge-assistant-grip')),
        column: column.map((c) => ({cls: c.className, box: box(c)})),
        corner: box(panel.querySelector('.forge-assistant-corner')),
        border: parseFloat(getComputedStyle(panel).borderLeftWidth) || 0,
    };
}"""


LAST_TAB = """() => {
    const nav = document.querySelector('#tabs > .tab-nav');
    const tabs = [...nav.querySelectorAll(':scope > button[role="tab"]')];
    const last = tabs[tabs.length - 1].getBoundingClientRect();
    const actions = document.querySelector('.mc-theme-actions').getBoundingClientRect();
    return {scrolled: nav.scrollTop, bottom: last.bottom, top: last.top, foot: actions.top, end: actions.bottom,
            selected: tabs[tabs.length - 1].classList.contains('selected')};
}"""


def geometry(page):
    return page.evaluate(GEOMETRY)


def transposed(page, tabs=None, panel=None):
    if tabs is not None:
        page.evaluate("(side) => forgeAssistant.shell.setTabRail(side)", tabs)
    if panel is not None:
        page.evaluate("(side) => forgeAssistant.shell.setPanelRail(side)", panel)
    settle(page, 6)


def open_panel(page):
    page.evaluate("forgeAssistant.shell.open()")
    settle(page, 8)


# --------------------------------------------------------------------------- #
# The tab bar
# --------------------------------------------------------------------------- #


class TestTheTabBarAsARail:
    @pytest.mark.parametrize("side", ["left", "right"])
    def test_a_slim_rail_down_the_side_tabs_first_and_the_other_buttons_apart_at_its_foot(self, browser, side):
        page = focus_page(browser, width=1280, height=800)
        transposed(page, tabs=side)
        found = geometry(page)
        page.close()

        nav, width, height = found["nav"], found["width"], found["height"]
        assert found["rail"] == side and found["position"] == "fixed"
        assert nav["top"] == 0 and nav["height"] == height and nav["width"] == RAIL
        assert (nav["left"] == 0) if side == "left" else (nav["right"] == width)
        tops = [tab["top"] for tab in found["tabs"]]
        assert tops == sorted(tops) and len({tab["left"] for tab in found["tabs"]}) == 1, "stacked, one above the next"
        assert found["tabs"][0]["top"] <= 8
        assert abs(found["actions"]["bottom"] - height) <= 1, "the other buttons stand at the rail's foot"
        assert found["actionsLine"] != "0px", "a line between the tabs and the other buttons"
        lefts = {round(kid["left"]) for kid in found["actionKids"]}
        assert len(found["actionKids"]) >= 3 and len(lefts) == 1, "stacked too"
        offsets = re.search(r"\) (-?[\d.]+)px (-?[\d.]+)px", found["accent"])
        assert found["accent"].endswith("inset") and offsets, found["accent"]
        assert (float(offsets.group(1)), float(offsets.group(2))) == ((-2, 0) if side == "left" else (2, 0)), \
            "the accent on the side that faces the page"
        assert found["ground"] == "none" and found["strips"] == "none", \
            "nothing of a bar's ground reaching across the page's padding"

    @pytest.mark.parametrize("side", ["left", "right"])
    def test_the_page_starts_at_the_windows_top_beside_the_rail(self, browser, side):
        page = focus_page(browser, width=1280, height=800)
        before = geometry(page)
        transposed(page, tabs=side)
        found = geometry(page)
        page.close()

        content, nav = found["content"], found["nav"]
        assert before["content"]["top"] > 40, "under the bar along the top before"
        assert content["top"] <= 8, "minimally padded: the page's own 8 px and no bar above it"
        if side == "left":
            assert content["left"] >= nav["right"] and content["left"] - nav["right"] <= 8
        else:
            assert content["right"] <= nav["left"] and nav["left"] - content["right"] <= 8

    def test_on_a_rail_the_tabs_are_icons_whatever_the_names_setting_says(self, browser):
        page = open_page(browser, assistant=USER_CSS_WHOLE, width=1280, height=800)
        settle(page, 8)
        names = geometry(page)
        transposed(page, tabs="left")
        railed = geometry(page)
        transposed(page, tabs="")
        back = geometry(page)
        page.close()

        assert names["icons"] is None and names["glyphs"] == 0
        assert railed["icons"] == "icons" and railed["glyphs"] >= 13
        assert back["rail"] is None and back["icons"] is None and back["glyphs"] == 0
        assert back["position"] == names["position"] == "sticky"
        assert back["nav"] == names["nav"], "along the top again, where it was"

    def test_the_selected_tab_is_brought_into_view_down_the_rail(self, browser):
        page = focus_page(browser, width=1280, height=420)
        transposed(page, tabs="left")
        page.evaluate("""() => {
            const tabs = [...document.querySelectorAll('#tabs > .tab-nav > button[role="tab"]')];
            tabs[tabs.length - 1].click();
        }""")
        page.evaluate("hooks.changed.forEach((fn) => fn())")
        settle(page, 8)
        found = page.evaluate(LAST_TAB)
        page.close()

        assert found["selected"] and found["scrolled"] > 0
        assert found["top"] >= 0 and found["bottom"] <= found["foot"], "clear of the buttons at the foot"
        assert abs(found["end"] - 420) <= 1, "the buttons stay at the foot while the tabs scroll"

    def test_a_tab_chosen_along_the_top_is_found_again_down_the_rail(self, browser):
        page = focus_page(browser, width=1280, height=420)
        page.evaluate("""() => {
            const tabs = [...document.querySelectorAll('#tabs > .tab-nav > button[role="tab"]')];
            tabs[tabs.length - 1].click();
        }""")
        page.evaluate("hooks.changed.forEach((fn) => fn())")
        settle(page, 8)
        transposed(page, tabs="left")
        found = page.evaluate(LAST_TAB)
        page.close()

        assert found["selected"] and found["scrolled"] > 0
        assert found["top"] >= 0 and found["bottom"] <= found["foot"]

    def test_the_wheel_over_the_rail_scrolls_it_down_even_when_a_theme_widens_a_tab(self, browser):
        """Along the top, a wheel over the icon bar scrolls it sideways. A
        rail scrolls down by itself, and the wheel is left to it -- also when
        a rule of a theme's makes a tab wider than the rail (Lobe's
        `min-width: fit-content !important` on every button is one such),
        which gives the rail a sideways overflow it clips."""
        page = focus_page(browser, width=1280, height=420)
        page.add_style_tag(content='#tabs > .tab-nav > button[role="tab"] { min-width: 72px !important; }')
        transposed(page, tabs="left")
        page.mouse.move(24, 150)
        page.mouse.wheel(0, 240)
        settle(page, 8)
        found = page.evaluate("""() => {
            const nav = document.querySelector('#tabs > .tab-nav');
            return {down: nav.scrollTop, across: nav.scrollLeft, wide: nav.scrollWidth > nav.clientWidth};
        }""")
        page.close()

        assert found["wide"] is True, "the stand-in rule gives the rail its sideways overflow"
        assert found["down"] > 0 and found["across"] == 0


DRAWERS = """() => {
    const box = (node) => {
        if (!node) return null;
        const r = node.getBoundingClientRect();
        return {left: r.left, top: r.top, right: r.right, bottom: r.bottom, width: r.width};
    };
    const quick = document.querySelector('#mc-theme-quick');
    return {quick: quick && !quick.hidden ? box(quick) : null,
            head: box(document.querySelector('#mc-theme-networks-txt2img')),
            page: box(document.querySelector('#txt2img_extra_tabs > .mc-theme-networks-shown')),
            nav: box(document.querySelector('#tabs > .tab-nav')),
            width: document.documentElement.clientWidth, height: innerHeight};
}"""


class TestTheDrawersBesideTheRail:
    def test_the_quick_settings_open_from_the_windows_top_beside_a_left_rail(self, browser):
        page = focus_page(browser, width=1280, height=800)
        transposed(page, tabs="left")
        page.click(".mc-theme-quick-toggle")
        settle(page, 6)
        found = page.evaluate(DRAWERS)
        page.close()

        quick = found["quick"]
        assert quick is not None
        assert quick["top"] == 0 and quick["left"] == RAIL and quick["bottom"] == found["height"]

    def test_the_extra_networks_open_from_the_windows_top_beside_a_right_rail_and_in_focus(self, browser):
        page = focus_page(browser, width=1280, height=800)
        transposed(page, tabs="right")
        page.click(".mc-theme-networks-toggle")
        settle(page, 8)
        found = page.evaluate(DRAWERS)
        enter_focus(page)
        focused = page.evaluate(DRAWERS)
        page.close()

        edge = found["width"] - RAIL
        for where in (found, focused):
            assert where["nav"]["left"] == edge
            assert where["head"]["top"] == 0 and where["head"]["right"] == edge, where["head"]
            assert where["page"]["right"] == edge and where["page"]["top"] >= where["head"]["bottom"] - 1


# --------------------------------------------------------------------------- #
# Focus mode
# --------------------------------------------------------------------------- #


class TestTheRailInFocusMode:
    @pytest.mark.parametrize("side", ["left", "right"])
    def test_the_kept_bar_is_the_rail_and_the_workspace_starts_at_the_top_beside_it(self, browser, side):
        page = focus_page(browser, width=1280, height=800)
        transposed(page, tabs=side)
        enter_focus(page)
        found = geometry(page)
        page.close()

        nav, root, width = found["nav"], found["root"], found["width"]
        assert found["focusRail"] == side
        assert nav["top"] == 0 and nav["height"] == found["height"] and nav["width"] == RAIL
        assert root["top"] == 0, "no bar above the workspace"
        if side == "left":
            assert nav["left"] == 0 and root["left"] == RAIL and root["right"] == width
        else:
            assert nav["right"] == width and root["right"] == width - RAIL and root["left"] == 0

    def test_hidden_it_goes_off_its_side_and_the_workspace_takes_the_window(self, browser):
        page = focus_page(browser, width=1280, height=800)
        transposed(page, tabs="left")
        page.evaluate("forgeAssistant.shell.setFocusBar(false)")
        enter_focus(page)
        hidden = geometry(page)
        page.evaluate("forgeAssistant.shell.setFocusBar(true)")
        settle(page, 6)
        shown = geometry(page)
        page.close()

        assert hidden["nav"]["right"] <= 0 and hidden["root"]["left"] == 0 and hidden["root"]["top"] == 0
        assert shown["nav"]["left"] == 0 and shown["root"]["left"] == RAIL

    def test_a_change_of_side_with_focus_on_moves_the_workspace_with_it(self, browser):
        """From the top to the left, across to the right -- the same rail,
        the same size, so nothing is resized to say so -- and back."""
        page = focus_page(browser, width=1280, height=800)
        enter_focus(page)
        along = geometry(page)
        transposed(page, tabs="left")
        left = geometry(page)
        transposed(page, tabs="right")
        right = geometry(page)
        transposed(page, tabs="")
        back = geometry(page)
        page.close()

        width = along["width"]
        assert along["root"]["top"] == 44 and along["focusRail"] is None
        assert left["root"]["top"] == 0 and left["root"]["left"] == RAIL and left["root"]["right"] == width
        assert right["root"]["top"] == 0 and right["root"]["left"] == 0
        assert right["root"]["right"] == width - RAIL and right["nav"]["right"] == width
        assert back["root"]["top"] == 44 and back["root"]["left"] == 0 and back["focusRail"] is None


# --------------------------------------------------------------------------- #
# The panel and the launcher beside the tab rail
# --------------------------------------------------------------------------- #

BOXES = """() => {
    const box = (node) => {
        if (!node || node.hidden) return null;
        const r = node.getBoundingClientRect();
        return {left: r.left, top: r.top, right: r.right, bottom: r.bottom, width: r.width, height: r.height};
    };
    return {panel: box(document.querySelector('.forge-assistant-panel')),
            launcher: box(document.querySelector('.forge-assistant-launcher')),
            corner: box(document.querySelector('.forge-assistant-corner')),
            where: document.querySelector('.forge-assistant-panel').getAttribute('data-corner'),
            width: document.documentElement.clientWidth, height: innerHeight};
}"""

GAP = 24           # the panel's gap from an edge on a wide window (forge_assistant.js, WIDE_GAP)


def anchored(page, panel, launcher=None):
    page.evaluate("""([panel, launcher]) => {
        const shell = forgeAssistant.shell;
        shell.state.panelAnchor = panel;
        shell.state.anchorOverride = launcher || panel;
        shell.placeNow();
    }""", [panel, launcher])
    settle(page, 4)


class TestThePanelBesideTheTabRail:
    """"The flyout menu is not properly handling the side tab bar either, lets
    make sure it does not get overlapped or render underneath side tab bar,
    only beside. I am able to resize it to beneath." The rail's width is room
    taken off its side of the window, as the safe-area insets are."""

    @pytest.mark.parametrize("side", ["left", "right"])
    def test_on_the_rails_side_the_panel_and_the_launcher_stand_beside_it(self, browser, side):
        page = focus_page(browser, width=1280, height=800)
        anchored(page, "top-" + side)
        transposed(page, tabs=side)
        open_panel(page)
        opened = page.evaluate(BOXES)
        page.evaluate("forgeAssistant.shell.close()")
        settle(page, 6)
        closed = page.evaluate(BOXES)
        page.close()

        width = opened["width"]
        for box in (opened["panel"], closed["launcher"]):
            if side == "left":
                assert box["left"] == RAIL + GAP
            else:
                # Placed at a whole pixel; a launcher's width is not one.
                assert abs(box["right"] - (width - RAIL - GAP)) < 1

    def test_the_corner_cannot_take_it_under_the_rail(self, browser):
        """Along the right with the rail on the left, the corner is on the
        panel's left; dragged to the window's edge it widens the panel only as
        far as the rail."""
        page = focus_page(browser, width=900, height=800)
        anchored(page, "top-right")
        transposed(page, tabs="left")
        open_panel(page)
        before = page.evaluate(BOXES)
        corner = before["corner"]
        x, y = corner["left"] + 3, corner["top"] + corner["height"] - 3
        page.mouse.move(x, y)
        page.mouse.down()
        page.mouse.move(2, y, steps=8)
        page.mouse.up()
        settle(page, 6)
        after = page.evaluate(BOXES)
        page.close()

        assert before["where"].endswith("left")
        assert after["panel"]["width"] > before["panel"]["width"] + 100, "it did widen"
        assert after["panel"]["left"] >= RAIL, "never under the rail"
        assert after["panel"]["right"] == before["panel"]["right"]

    def test_the_edge_strip_cannot_either(self, browser):
        page = focus_page(browser, width=900, height=800)
        anchored(page, "top-right")
        transposed(page, tabs="left")
        open_panel(page)
        strip = page.evaluate("""() => { const r = document.querySelector('.forge-assistant-resize')
            .getBoundingClientRect(); return {x: r.left + r.width / 2, y: r.top + r.height / 2}; }""")
        page.mouse.move(strip["x"], strip["y"])
        page.mouse.down()
        page.mouse.move(1, strip["y"], steps=8)
        page.mouse.up()
        settle(page, 6)
        after = page.evaluate(BOXES)
        page.close()

        assert after["panel"]["left"] >= RAIL

    def test_a_drag_against_the_rail_stops_beside_it(self, browser):
        page = focus_page(browser, width=1280, height=800)
        page.evaluate("forgeAssistant.shell.setFreeFloat(true)")
        transposed(page, tabs="left")
        open_panel(page)
        grip = page.evaluate("""() => { const r = document.querySelector('.forge-assistant-grip')
            .getBoundingClientRect(); return {x: r.left + r.width / 2, y: r.top + r.height / 2}; }""")
        page.mouse.move(grip["x"], grip["y"])
        page.mouse.down()
        page.mouse.move(grip["x"] - 2000, grip["y"], steps=10)
        moving = page.evaluate(BOXES)
        page.mouse.up()
        settle(page, 6)
        let_go = page.evaluate(BOXES)
        page.close()

        assert moving["panel"]["left"] == RAIL + GAP, "drawn beside it under the hand"
        assert let_go["panel"]["left"] == RAIL + GAP, "and put down there"

    def test_the_rail_coming_and_going_moves_the_panel_beside_it_and_back(self, browser):
        page = focus_page(browser, width=1280, height=800)
        anchored(page, "bottom-left")
        open_panel(page)
        along = page.evaluate(BOXES)
        transposed(page, tabs="left")
        beside = page.evaluate(BOXES)
        transposed(page, tabs="right")
        across = page.evaluate(BOXES)
        transposed(page, tabs="")
        back = page.evaluate(BOXES)
        page.close()

        assert along["panel"]["left"] == GAP
        assert beside["panel"]["left"] == RAIL + GAP, "placed again, with nothing else asking"
        assert across["panel"]["left"] == GAP
        assert back["panel"]["left"] == GAP

    def test_in_focus_mode_beside_the_kept_rail_and_at_the_edge_while_it_is_hidden(self, browser):
        page = focus_page(browser, width=1280, height=800)
        anchored(page, "top-left")
        transposed(page, tabs="left")
        enter_focus(page)
        open_panel(page)
        shown = page.evaluate(BOXES)
        page.evaluate("forgeAssistant.shell.setFocusBar(false)")
        settle(page, 6)
        hidden = page.evaluate(BOXES)
        page.close()

        assert shown["panel"]["left"] == RAIL + GAP
        assert hidden["panel"]["left"] == GAP, "the workspace has the window, and so does the panel"

    @pytest.mark.parametrize("side", ["left", "right"])
    def test_the_phones_sheet_stands_beside_it(self, browser, side):
        page = focus_page(browser, width=390, height=760)
        transposed(page, tabs=side)
        open_panel(page)
        found = page.evaluate(BOXES)
        sheet = page.evaluate("document.querySelector('.forge-assistant-panel')"
                              ".classList.contains('forge-assistant-sheet')")
        page.close()

        assert sheet is True
        if side == "left":
            assert found["panel"]["left"] == RAIL and found["panel"]["right"] == found["width"]
        else:
            assert found["panel"]["left"] == 0 and found["panel"]["right"] == found["width"] - RAIL


# --------------------------------------------------------------------------- #
# The panel's buttons
# --------------------------------------------------------------------------- #


class TestThePanelsButtonsAsARail:
    @pytest.mark.parametrize("side", ["left", "right"])
    def test_down_the_panels_side_with_its_column_starting_at_its_top(self, browser, side):
        page = focus_page(browser, width=1280, height=800)
        open_panel(page)
        before = page.evaluate(PANEL)
        transposed(page, panel=side)
        found = page.evaluate(PANEL)
        page.close()

        panel, header = found["panel"], found["header"]
        inner = panel["height"] - 2 * found["border"]
        assert found["rail"] == side
        assert header["width"] == PANEL_RAIL and abs(header["height"] - inner) <= 1, "the panel's height"
        assert (header["left"] == panel["left"] + found["border"]) if side == "left" \
            else (header["right"] == panel["right"] - found["border"])
        lefts = {round(button["left"]) for button in found["buttons"]}
        tops = [button["top"] for button in found["buttons"]]
        assert len(lefts) == 1 and tops == sorted(tops), "stacked down the rail"
        first = min(found["column"], key=lambda part: part["box"]["top"])["box"]
        assert before["column"] and min(p["box"]["top"] for p in before["column"]) - before["panel"]["top"] > 30
        assert first["top"] - panel["top"] <= 4, "the column starts at the panel's top"
        for part in found["column"]:
            box = part["box"]
            if box["width"] == 0:
                continue
            if side == "left":
                assert box["left"] >= header["right"] - 0.5, part["cls"]
            else:
                assert box["right"] <= header["left"] + 0.5, part["cls"]

    def test_the_grip_runs_down_the_rail_and_the_panel_is_dragged_by_it(self, browser):
        page = focus_page(browser, width=1280, height=800)
        open_panel(page)
        transposed(page, panel="left")
        found = page.evaluate(PANEL)
        grip = found["grip"]
        x, y = grip["left"] + grip["width"] / 2, grip["top"] + grip["height"] / 2
        page.mouse.move(x, y)
        page.mouse.down()
        # Measured under the drag: let go anchored, the panel goes back to its
        # resting place, which a move this short does not change.
        page.mouse.move(x - 120, y - 80, steps=6)
        moving = page.evaluate(PANEL)
        page.mouse.up()
        settle(page, 8)
        page.close()

        assert grip["height"] > grip["width"] and grip["width"] <= PANEL_RAIL
        # Bottom-justified, at the user's word ("when set to side instead of
        # top, to be bottom justified"): the grip first, from the rail's top,
        # and the buttons stacked at its foot.
        header = found["header"]
        assert grip["top"] - header["top"] <= 8, "the grip first, at the rail's top"
        assert grip["bottom"] <= found["buttons"][0]["top"] + 1, "above the buttons"
        assert header["bottom"] - found["buttons"][-1]["bottom"] <= 8, "the buttons at the rail's foot"
        assert moving["panel"]["left"] == found["panel"]["left"] - 120, "dragged by the rail's grip"
        assert moving["panel"]["top"] == found["panel"]["top"] - 80

    def test_a_panel_shorter_than_its_rail_is_as_tall_as_the_rail(self, browser):
        page = focus_page(browser, width=1280, height=800)
        open_panel(page)
        page.evaluate("forgeAssistant.shell.toggleChat ? forgeAssistant.shell.toggleChat() : null")
        page.evaluate("(() => { const s = forgeAssistant.shell; if (s.state.conversationExpanded) "
                      "s.nodes.chat.click(); })()")
        settle(page, 6)
        transposed(page, panel="left")
        found = page.evaluate(PANEL)
        page.close()

        last = found["buttons"][-1]
        assert last["bottom"] <= found["header"]["bottom"] + 0.5, "every button whole"
        assert found["grip"]["height"] >= 3.4 * 14, "and the grip its least height"

    def test_the_edge_strip_leaves_the_rails_buttons_to_their_presses(self, browser):
        """The panel widens by an 8 px strip on its left edge; down a rail on
        that side, the strip starts where the rail ends, so a press on a
        button's edge is the button's."""
        page = focus_page(browser, width=1280, height=800)
        open_panel(page)
        transposed(page, panel="left")
        found = page.evaluate("""() => {
            const header = document.querySelector('.forge-assistant-panel .forge-assistant-header');
            const buttons = [...header.children].filter((c) => c.tagName === 'BUTTON' && !c.hidden
                && getComputedStyle(c).display !== 'none');
            const strip = document.querySelector('.forge-assistant-panel > .forge-assistant-resize');
            return {answers: buttons.map((b) => {
                        const r = b.getBoundingClientRect();
                        const at = document.elementFromPoint(r.left + 1, r.top + r.height / 2);
                        return !!at && (at === b || b.contains(at));
                    }),
                    strip: strip ? strip.getBoundingClientRect().left : null,
                    stripTop: strip ? strip.getBoundingClientRect().top : null,
                    top: header.getBoundingClientRect().top,
                    rail: header.getBoundingClientRect().right};
        }""")
        page.close()

        assert found["answers"] and all(found["answers"]), found
        assert found["strip"] is not None and found["strip"] >= found["rail"] - 1
        assert found["stripTop"] == found["top"], "the column's whole height: no header row above it"

    def test_the_memory_reading_runs_down_the_grip(self, browser):
        page = focus_page(browser, width=1280, height=800)
        open_panel(page)
        transposed(page, panel="right")
        page.evaluate("""() => {
            const reading = document.querySelector('.forge-assistant-panel .forge-assistant-memory');
            reading.hidden = false;
            reading.textContent = 'RAM 31.2 GB \u00b7 VRAM 11.4 GB';
        }""")
        settle(page, 4)
        found = page.evaluate("""() => {
            const r = document.querySelector('.forge-assistant-panel .forge-assistant-memory').getBoundingClientRect();
            const g = document.querySelector('.forge-assistant-panel .forge-assistant-grip').getBoundingClientRect();
            return {width: r.width, height: r.height, gripLeft: g.left, gripRight: g.right,
                    left: r.left, right: r.right};
        }""")
        page.close()

        assert found["height"] > found["width"] > 0, "written down the rail, not across it"
        assert found["width"] <= PANEL_RAIL
        assert found["left"] >= found["gripLeft"] - 0.5 and found["right"] <= found["gripRight"] + 0.5

    def test_a_switch_of_workspace_that_adds_buttons_keeps_every_one_whole(self, browser):
        """The rail's buttons follow the workspace: ▶ and 🎛 show on Txt2Img and
        not on Extras. A short panel that came to Txt2Img from Extras is made
        tall enough again for the rail it has now."""
        page = focus_page(browser, width=1280, height=800)
        page.click("#tab_extras-button")
        page.evaluate("hooks.changed.forEach((fn) => fn())")
        settle(page, 6)
        open_panel(page)
        page.evaluate("(() => { const s = forgeAssistant.shell; if (s.state.conversationExpanded) "
                      "s.nodes.chat.click(); })()")
        settle(page, 6)
        transposed(page, panel="left")
        extras = page.evaluate(PANEL)
        # From the panel's own workspace menu: a press on the page's tab bar
        # with the panel open only puts the panel away (Tap Outside to Close).
        page.evaluate("forgeAssistant.shell.switchWorkspace('tab_txt2img')")
        page.evaluate("hooks.changed.forEach((fn) => fn())")
        settle(page, 8)
        txt2img = page.evaluate(PANEL)
        page.close()

        assert len(txt2img["buttons"]) > len(extras["buttons"]), "more buttons on Txt2Img"
        assert extras["panel"]["height"] < extras["header"]["height"] + 200, "a short panel"
        last = txt2img["buttons"][-1]
        assert last["bottom"] <= txt2img["header"]["bottom"] + 0.5, "every button whole"
        assert txt2img["grip"]["height"] >= 3.4 * 14, "and the grip its least height"

    def test_the_corner_sits_beside_the_rail_not_on_it(self, browser):
        page = focus_page(browser, width=1280, height=800)
        open_panel(page)
        transposed(page, panel="left")
        found = page.evaluate(PANEL)
        corner_side = page.evaluate("document.querySelector('.forge-assistant-panel').getAttribute('data-corner')")
        page.close()

        if corner_side and corner_side.endswith("left"):
            assert found["corner"]["left"] >= found["header"]["right"] - 0.5
        else:
            pytest.skip("the panel's corner is on the other side here")


# --------------------------------------------------------------------------- #
# The ⋯ menu
# --------------------------------------------------------------------------- #

MENU = """(which) => {
    const shell = forgeAssistant.shell;
    shell.showMenu(which);
    const items = [...shell.nodes.menu.children].map((item) => ({
        text: item.textContent, role: item.getAttribute('role'),
        checked: item.getAttribute('aria-checked'), disabled: !!item.disabled,
        rail: item.dataset ? item.dataset.rail || null : null}));
    shell.closeMenu();
    return items;
}"""


class TestTheMenu:
    def test_transpose_view_is_in_the_panel_group_and_opens_the_two_choices(self, browser):
        page = focus_page(browser, width=1280, height=800)
        open_panel(page)
        utilities = [item["text"] for item in page.evaluate(MENU, "utilities")]
        choices = page.evaluate(MENU, "transpose")
        page.close()

        assert utilities.index("Transpose View") == utilities.index("Tap Outside to Close") + 1
        texts = [item["text"] for item in choices if item["role"] != "menuitem" or item["rail"]]
        assert texts[:8] == ["Tab Bar", "Top", "Left", "Right", "Panel Buttons", "Top", "Left", "Right"]
        radios = [item for item in choices if item["role"] == "menuitemradio"]
        assert [item["checked"] for item in radios] == ["true", "false", "false"] * 2
        assert not any(item["disabled"] for item in radios)

    def test_a_choice_is_made_from_the_menu_and_says_which_is_on(self, browser):
        page = focus_page(browser, width=1280, height=800)
        open_panel(page)
        page.evaluate("""() => {
            const shell = forgeAssistant.shell;
            shell.showMenu('transpose');
            [...shell.nodes.menu.children].find((i) => i.dataset && i.dataset.rail === 'tabs'
                && i.dataset.side === 'right').click();
            shell.showMenu('transpose');
            [...shell.nodes.menu.children].find((i) => i.dataset && i.dataset.rail === 'panel'
                && i.dataset.side === 'left').click();
        }""")
        settle(page, 6)
        choices = page.evaluate(MENU, "transpose")
        found = geometry(page)
        panel = page.evaluate(PANEL)
        page.close()

        assert found["rail"] == "right" and panel["rail"] == "left"
        radios = [item["checked"] for item in choices if item["role"] == "menuitemradio"]
        assert radios == ["false", "false", "true", "false", "true", "false"]

    def test_the_choices_are_kept_in_this_browser_and_the_bar_is_never_drawn_along_the_top_first(self, browser):
        first = focus_page(browser, width=1280, height=800)
        transposed(first, tabs="left", panel="right")
        context = first.context
        second = open_page(browser, ICONS, context=context, assistant=USER_CSS_WHOLE, loaded=False)
        early = second.evaluate("document.documentElement.getAttribute('data-forge-assistant-rail')")
        second.evaluate("hooks.loaded.forEach((fn) => fn())")
        settle(second, 8)
        second.evaluate("forgeAssistant.shell.open()")
        settle(second, 6)
        found = geometry(second)
        panel = second.evaluate(PANEL)
        second.close()
        first.close()

        assert early == "left", "on the root before the page's UI is built"
        assert found["rail"] == "left" and found["nav"]["left"] == 0 and found["nav"]["width"] == RAIL
        assert panel["rail"] == "right"

    def test_without_the_theme_the_tab_bar_choices_wait_for_it(self, browser):
        page = open_page(browser, {"model_chain_theme": False}, assistant=USER_CSS_WHOLE,
                         width=1280, height=800)
        settle(page, 8)
        page.evaluate("forgeAssistant.shell.open()")
        settle(page, 6)
        choices = page.evaluate(MENU, "transpose")
        page.close()

        tabs = [item for item in choices if item["rail"] == "tabs"]
        panel = [item for item in choices if item["rail"] == "panel"]
        assert [item["disabled"] for item in tabs] == [False, True, True]
        assert not any(item["disabled"] for item in panel)
