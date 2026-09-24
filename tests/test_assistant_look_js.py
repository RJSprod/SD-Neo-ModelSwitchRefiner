"""The launcher's look: validated, applied, edited, and cheap to animate.

Asked for: "in '…' menu, add a customize menu. It should open a pop up with
customize options ... background color, title, corner radius, font color,
stroke, the same animation that we have for our custom progress bar always
playing on the button ... a nice touch friendly place to decide."

The look lives in ``localStorage``, so every test of reading it back is a test
of what happens when somebody -- or something -- has put anything at all
there. And because the animation is always playing on the one control that is
always on screen, the stylesheet is held to animating nothing but compositor
properties.

Every test here was checked against the change it guards by reverting that
change and watching the test fail.
"""

from __future__ import annotations

import pathlib
import re

from test_assistant_js import LOOK, SHELL, run

ROOT = pathlib.Path(__file__).resolve().parent.parent


def stylesheet() -> str:
    return (ROOT / "style.css").read_text(encoding="utf-8")


# A DOM a little richer than the harness's: children that can be found by
# class, inserted before one another and removed, which is all `decorate`
# does with them.
RICH = """
const created = document.createElement;
function enrich(node) {
    node.parentNode = null;
    node.remove = () => {
        if (node.parentNode) {
            node.parentNode.children = node.parentNode.children.filter((c) => c !== node);
        }
        node.parentNode = null;
    };
    node.appendChild = (child) => {
        node.children.push(child);
        child.parentNode = node;
        return child;
    };
    node.insertBefore = (child, before) => {
        const at = node.children.indexOf(before);
        if (at < 0) node.children.push(child);
        else node.children.splice(at, 0, child);
        child.parentNode = node;
        return child;
    };
    const matches = (child, selector) => (" " + child.className + " ")
        .indexOf(" " + selector.replace(/^\\./, "") + " ") >= 0;
    const walk = (from, selector, direct, found) => {
        from.children.forEach((child) => {
            if (child && child.className !== undefined && matches(child, selector)) found.push(child);
            if (!direct && child && child.children) walk(child, selector, direct, found);
        });
        return found;
    };
    node.querySelectorAll = (selector) => {
        const direct = selector.indexOf(":scope > ") === 0;
        return walk(node, selector.replace(":scope > ", ""), direct, []);
    };
    node.querySelector = (selector) => node.querySelectorAll(selector)[0] || null;
    return node;
}
document.createElement = (tag) => enrich(created(tag));
const LOOK = NS.look;
"""


class TestAStoredLookIsCheckedOnTheWayIn:
    def test_anything_that_is_not_a_look_is_the_default(self):
        found = run(RICH + """
            const d = LOOK.sanitize(LOOK.DEFAULTS);
            const cases = [null, "a string", 42, [], {version: 1}];
            console.log(JSON.stringify({same: cases.map((raw) =>
                JSON.stringify(LOOK.sanitize(raw)) === JSON.stringify(d))}));
        """, sources=("look",))

        assert found["same"] == [True] * 5

    def test_colours_are_six_digit_hex_or_nothing(self):
        """Nothing read from storage reaches a style unchecked -- not a named
        colour, not a url(), not a var() somebody typed."""
        found = run(RICH + """
            const got = ["#ABCDEF", "red", "url(javascript:alert(1))", "#abc",
                         "var(--x)", "#12345g", " #00ff00 "]
                .map((fill) => LOOK.sanitize({fill}).fill);
            console.log(JSON.stringify({got, text: LOOK.sanitize({text: "auto"}).text,
                                        textBad: LOOK.sanitize({text: "blue"}).text}));
        """, sources=("look",))

        assert found["got"] == ["#abcdef", "", "", "", "", "", "#00ff00"]
        assert found["text"] == "auto"
        assert found["textBad"] == "auto", "an unreadable choice falls back to Auto"

    def test_numbers_and_choices_are_held_to_their_range(self):
        found = run(RICH + """
            const s = (raw) => LOOK.sanitize(raw);
            console.log(JSON.stringify({
                pill: s({radius: 1e9}).radius, negative: s({radius: -5}).radius,
                rounded: s({radius: 13.7}).radius, nan: s({radius: "x"}).radius,
                stroke: s({strokeWidth: 99}).strokeWidth, angle: s({angle: 720}).angle,
                motion: s({motion: "explode"}).motion, icon: s({icon: "<img src=x>"}).icon,
                size: s({size: "huge"}).size, speed: s({speed: "ludicrous"}).speed,
                anyway: s({anyway: "yes"}).anyway, really: s({anyway: true}).anyway}));
        """, sources=("look",))

        assert found == {"pill": 999, "negative": 0, "rounded": 14, "nan": 10, "stroke": 4,
                         "angle": 360, "motion": "none", "icon": "", "size": "regular",
                         "speed": "normal", "anyway": False, "really": True}

    def test_a_title_is_printable_text_and_capped(self):
        found = run(RICH + """
            const long = "x".repeat(60);
            const emoji = "\\ud83d\\udd25".repeat(45);
            console.log(JSON.stringify({
                controls: LOOK.sanitize({title: "a\\u0000b\\nc\\u2028d"}).title,
                trimmed: LOOK.sanitize({title: "  Studio  "}).title,
                long: LOOK.sanitize({title: long}).title.length,
                emoji: Array.from(LOOK.sanitize({title: emoji}).title).length}));
        """, sources=("look",))

        assert found["controls"] == "abcd"
        assert found["trimmed"] == "Studio"
        assert found["long"] == 40
        assert found["emoji"] == 40, "capped by characters, never mid-way through one"

    def test_storage_round_trips_and_forgives(self):
        found = run(RICH + """
            const box = {};
            globalThis.localStorage = {getItem: (k) => box[k] === undefined ? null : box[k],
                                       setItem: (k, v) => { box[k] = String(v); }};
            const neon = LOOK.withPreset("neon");
            const saved = LOOK.save(neon);
            const back = LOOK.load();
            box[LOOK.key()] = "{not json";
            const corrupt = LOOK.load().motion;
            box[LOOK.key()] = JSON.stringify(Object.assign({}, neon, {version: 99}));
            const future = LOOK.load().motion;
            globalThis.localStorage = {getItem() { throw new Error("blocked"); },
                                       setItem() { throw new Error("blocked"); }};
            const blocked = LOOK.load().motion;
            const refused = LOOK.save(neon);
            console.log(JSON.stringify({saved, same: JSON.stringify(back) === JSON.stringify(neon),
                                        corrupt, future, blocked, refused}));
        """, sources=("look",))

        assert found["saved"] is True
        assert found["same"] is True
        assert found["corrupt"] == "none"
        assert found["future"] == "none", "a version this build does not know is not guessed at"
        assert found["blocked"] == "none"
        assert found["refused"] is False

    def test_nothing_is_ever_written_as_markup(self):
        """A title is text. The module has no reason to parse HTML anywhere."""
        source = LOOK.read_text(encoding="utf-8")

        assert "innerHTML" not in source
        assert "insertAdjacentHTML" not in source
        assert "outerHTML" not in source


class TestWhatALookResolvesTo:
    def test_the_default_look_leaves_the_theme_in_charge(self):
        """An untouched launcher must be exactly what it was."""
        found = run(RICH + """
            const r = LOOK.resolve(LOOK.DEFAULTS, {label: "Forge Assistant", appearance: "text"});
            console.log(JSON.stringify({vars: Object.keys(r.vars), motion: r.motion,
                                        parts: r.parts, label: r.label}));
        """, sources=("look",))

        assert "--fa-look-fill" not in found["vars"]
        assert "--fa-look-text" not in found["vars"]
        assert "--fa-look-stroke" not in found["vars"]
        assert found["motion"] == "none"
        assert found["parts"] == []
        assert found["label"] == "Forge Assistant"

    def test_two_fill_stops_are_a_gradient(self):
        found = run(RICH + """
            const r = LOOK.resolve({fill: "#ff0000", fill2: "#0000ff", angle: 90}, {});
            console.log(JSON.stringify({fill: r.vars["--fa-look-fill"]}));
        """, sources=("look",))

        assert found["fill"] == "linear-gradient(90deg, #ff0000, #0000ff)"

    def test_auto_text_reads_against_the_worse_stop(self):
        found = run(RICH + """
            const t = (look) => LOOK.autoText(LOOK.sanitize(look));
            console.log(JSON.stringify({
                light: t({fill: "#ffffff"}), dark: t({fill: "#111827"}),
                mixed: t({fill: "#ffffff", fill2: "#000000"}),
                theme: t({})}));
        """, sources=("look",))

        assert found["light"] == "#000000"
        assert found["dark"] == "#ffffff"
        assert found["mixed"] in ("#000000", "#ffffff")
        assert found["theme"] == "", "no fill chosen, so the theme's text colour stands"

    def test_ooze_is_read_against_the_sludge_not_the_fill(self):
        """The first Toxic preset put lime text on a dark fill, and Ooze then
        painted bright green sludge over the fill: 1.4:1."""
        found = run(RICH + """
            const look = LOOK.sanitize({fill: "#0d2b0f", accent: "#66ff33", motion: "ooze",
                                        text: "#d9ff66"});
            const still = LOOK.sanitize(Object.assign({}, look, {motion: "none"}));
            console.log(JSON.stringify({ooze: LOOK.legibility(look), still: LOOK.legibility(still),
                                        auto: LOOK.autoText(look)}));
        """, sources=("look",))

        assert found["ooze"] < 3, "measured against what is really behind the title"
        assert found["still"] > 7
        assert found["auto"] == "#000000"

    def test_every_preset_can_be_read(self):
        """A guard for the next preset somebody adds."""
        found = run(RICH + """
            console.log(JSON.stringify({ratios: LOOK.PRESETS.map((preset) =>
                [preset.id, LOOK.legibility(LOOK.withPreset(preset.id))])}));
        """, sources=("look",))

        for preset, ratio in found["ratios"]:
            assert ratio is None or ratio >= 4.5, (preset, ratio)

    def test_contrast_is_the_wcag_ratio(self):
        found = run(RICH + """
            console.log(JSON.stringify({max: LOOK.contrast("#000000", "#ffffff"),
                                        none: LOOK.contrast("#777777", "#777777")}));
        """, sources=("look",))

        assert round(found["max"], 2) == 21
        assert found["none"] == 1

    def test_match_follows_the_progress_bar_settings(self):
        found = run(RICH + """
            const follow = (settings) => {
                globalThis.opts = settings;
                return LOOK.resolve({motion: "match"}, {}).motion;
            };
            console.log(JSON.stringify({
                off: follow({model_chain_style_enable: false, model_chain_style_theme: "Ooze"}),
                ooze: follow({model_chain_style_enable: true, model_chain_style_theme: "Ooze"}),
                neon: follow({model_chain_style_enable: true, model_chain_style_theme: "Neon"}),
                flat: follow({model_chain_style_enable: true, model_chain_style_theme: "Flat"}),
                custom: follow({model_chain_style_enable: true, model_chain_style_theme: "Custom",
                                model_chain_style_sheen: true, model_chain_style_glow: true}),
                odd: follow({model_chain_style_enable: true, model_chain_style_theme: "Plasma"}),
                none: (delete globalThis.opts, LOOK.resolve({motion: "match"}, {}).motion),
                accent: LOOK.resolve({motion: "match"}, {}).vars["--fa-look-accent"]}));
        """, sources=("look",))

        assert found["off"] == "none", "a progress bar with no styling has no animation"
        assert found["ooze"] == "ooze"
        assert found["neon"] == "neon"
        assert found["flat"] == "none"
        assert found["custom"] == "neon"
        assert found["odd"] == "none"
        assert found["none"] == "none"
        assert "--mc-progress-fill" in found["accent"], "and in the progress bar's colour"

    def test_the_title_and_icon_override_settings_only_when_chosen(self):
        found = run(RICH + """
            const base = {label: "Forge Assistant", appearance: "text"};
            const plain = LOOK.resolve({}, base);
            const titled = LOOK.resolve({title: "Studio", icon: "\\u2726"}, base);
            const iconOnly = LOOK.resolve({}, {label: "X", appearance: "icon"});
            console.log(JSON.stringify({plain: [plain.label, plain.showIcon],
                                        titled: [titled.label, titled.showIcon, titled.icon],
                                        iconOnly: [iconOnly.showLabel, iconOnly.showIcon]}));
        """, sources=("look",))

        assert found["plain"] == ["Forge Assistant", False]
        assert found["titled"] == ["Studio", True, "✦"]
        assert found["iconOnly"] == [False, True]

    def test_speed_scales_every_animation_the_same_way(self):
        found = run(RICH + """
            console.log(JSON.stringify(["slow", "normal", "fast"].map((speed) =>
                LOOK.resolve({speed}, {}).vars["--fa-look-speed"])));
        """, sources=("look",))

        assert [float(value) for value in found] == [1.6, 1.0, 0.6]


class TestApplyingALook:
    def test_each_animation_gets_its_layers_and_loses_them_after(self):
        found = run(RICH + """
            const node = document.createElement("button");
            const names = () => {
                const all = [];
                const walk = (n) => n.children.forEach((c) => {
                    all.push(c.className.split(" ").pop()); if (c.children) walk(c); });
                walk(node);
                return all.sort();
            };
            const seen = {};
            ["sheen", "pulse", "neon", "ooze", "aurora", "none"].forEach((motion) => {
                LOOK.apply(node, {motion}, {});
                seen[motion] = {layers: names(), motion: node.getAttribute("data-look-motion")};
            });
            // Ooze to Pulse keeps the air layer, so the bubbles have to be
            // taken out of it one by one rather than going with it.
            LOOK.apply(node, {motion: "ooze"}, {});
            LOOK.apply(node, {motion: "pulse"}, {});
            seen.oozeThenPulse = {layers: names()};
            console.log(JSON.stringify(seen));
        """, sources=("look",))

        assert found["sheen"]["layers"] == ["forge-assistant-look-clip"]
        assert found["pulse"]["layers"] == ["forge-assistant-look-air",
                                            "forge-assistant-look-halo"]
        assert found["neon"]["motion"] == "sheen pulse lit"
        assert found["ooze"]["layers"].count("forge-assistant-look-bubble") == 7
        assert "forge-assistant-look-clip" in found["ooze"]["layers"]
        assert found["aurora"]["layers"] == ["forge-assistant-look-clip"]
        assert found["none"] == {"layers": [], "motion": None}, (
            "nothing left behind when the animation goes")
        assert found["oozeThenPulse"]["layers"] == ["forge-assistant-look-air",
                                                     "forge-assistant-look-halo"]

    def test_a_colour_taken_away_is_taken_off_the_element(self):
        found = run(RICH + """
            const node = document.createElement("button");
            LOOK.apply(node, {fill: "#ff0000", text: "#ffffff", stroke: "#00ff00"}, {});
            const before = node.style.getPropertyValue("--fa-look-fill");
            LOOK.apply(node, {}, {});
            console.log(JSON.stringify({before,
                after: node.style.getPropertyValue("--fa-look-fill"),
                text: node.style.getPropertyValue("--fa-look-text"),
                stroke: node.style.getPropertyValue("--fa-look-stroke")}));
        """, sources=("look",))

        assert found["before"] == "#ff0000"
        assert found == {"before": "#ff0000", "after": "", "text": "", "stroke": ""}

    def test_the_title_is_set_as_text(self):
        found = run(RICH + """
            const node = document.createElement("button");
            const label = document.createElement("span");
            const icon = document.createElement("span");
            LOOK.apply(node, {title: "<b>bold</b>"}, {label: "L", appearance: "text"},
                       {label, icon});
            console.log(JSON.stringify({text: label.textContent, html: label.innerHTML}));
        """, sources=("look",))

        assert found["text"] == "<b>bold</b>"
        assert found["html"] == ""

    def test_the_system_preference_is_carried_unless_overridden(self):
        found = run(RICH + """
            const node = document.createElement("button");
            LOOK.apply(node, {motion: "sheen"}, {});
            const plain = node.hasAttribute("data-look-anyway");
            LOOK.apply(node, {motion: "sheen", anyway: true}, {});
            console.log(JSON.stringify({plain, anyway: node.hasAttribute("data-look-anyway")}));
        """, sources=("look",))

        assert found == {"plain": False, "anyway": True}


class TestTheStylesheetAnimatesOnTheCompositor:
    """The launcher is always on screen and these always run. Every keyframe
    they use may change only what the compositor can change on its own."""

    def keyframes(self):
        css = stylesheet()
        blocks = {}
        for name in re.findall(r"@keyframes (fa-look-[\w-]+)", css):
            body = css.split("@keyframes " + name + " {", 1)[1]
            depth, end = 1, 0
            for index, char in enumerate(body):
                depth += char == "{"
                depth -= char == "}"
                if depth == 0:
                    end = index
                    break
            blocks[name] = body[:end]
        return blocks

    def test_every_keyframe_changes_only_transform_and_opacity(self):
        blocks = self.keyframes()

        assert set(blocks) >= {"fa-look-sheen", "fa-look-pulse", "fa-look-skin",
                               "fa-look-spin"}
        for name, body in blocks.items():
            properties = set(re.findall(r"([a-z-]+)\s*:", body))
            assert properties <= {"transform", "opacity", "rotate", "translate", "scale"}, (
                name, properties)

    def test_the_bubbles_are_the_progress_bars_own(self):
        """"The same animation": the launcher's bubbles use the progress bar's
        class and keyframes, not a copy that can drift from them."""
        source = LOOK.read_text(encoding="utf-8")

        assert '"mc-ooze-bubble forge-assistant-look-bubble"' in source
        css = stylesheet()
        rise = css.split("@keyframes mc-ooze-rise {", 1)[1].split("\n}", 1)[0]
        assert set(re.findall(r"([a-z-]+)\s*:", rise)) <= {"transform", "opacity"}

    def test_reduced_motion_is_honoured_unless_the_dialog_was_told_otherwise(self):
        css = stylesheet()
        block = css.split("@media (prefers-reduced-motion: reduce) {\n    .forge-assistant-launcher[data-look-motion]", 1)[1] \
            .split("\n}\n", 1)[0]

        assert ":not([data-look-anyway])" in block
        assert "animation: none" in block
        assert ".forge-assistant-look-bubble" in block and "display: none" in block

    def test_the_layers_never_take_a_press(self):
        css = stylesheet()
        rule = css.split(".forge-assistant-look-clip,\n.forge-assistant-look-air {", 1)[1] \
            .split("}", 1)[0]

        assert "pointer-events: none" in rule
        assert "z-index: -1" in rule, "behind the title"

    def test_every_control_in_the_dialog_is_a_finger_wide(self):
        css = stylesheet()
        chip = css.split(".forge-assistant-look-chip,\n.forge-assistant-look-action {", 1)[1] \
            .split("}", 1)[0]
        swatch = css.split(".forge-assistant-look-swatch {", 1)[1].split("}", 1)[0]
        title = css.split(".forge-assistant-look-title {", 1)[1].split("}", 1)[0]

        assert "min-height: 44px" in chip and "min-width: 44px" in chip
        assert "width: 44px" in swatch and "height: 44px" in swatch
        assert "min-height: 44px" in title
        assert "max(16px" in title, "or iOS zooms the page when the title field is tapped"


EDITOR = RICH + """
const box = {};
globalThis.localStorage = {getItem: (k) => box[k] === undefined ? null : box[k],
                           setItem: (k, v) => { box[k] = String(v); }};
const applied = [];
const shell = {settings: {label: "Forge Assistant", appearance: "text"},
               nodes: {root: document.createElement("div"),
                       launcher: document.createElement("button")},
               applyLook(look) { applied.push(look ? look.motion + "/" + look.title : "saved"); }};
shell.nodes.launcher.hidden = false;
const editor = LOOK.editor(shell);
globalThis.matchMedia = () => ({matches: false});
"""


class TestTheEditor:
    def test_it_is_built_on_first_use_and_not_before(self):
        found = run(EDITOR + """
            const before = !!editor.dialog;
            editor.open();
            console.log(JSON.stringify({before, after: !!editor.dialog,
                                        inRoot: shell.nodes.root.children.includes(editor.dialog)}));
        """, sources=("look",))

        assert found == {"before": False, "after": True, "inRoot": True}

    def test_every_change_is_live_on_the_real_launcher(self):
        found = run(EDITOR + """
            editor.open();
            applied.length = 0;
            editor.set({motion: "pulse"});
            editor.set({title: "Studio"});
            console.log(JSON.stringify({applied}));
        """, sources=("look",))

        assert found["applied"] == ["pulse/", "pulse/Studio"]

    def test_cancel_puts_it_back_and_save_keeps_it(self):
        found = run(EDITOR + """
            LOOK.save(LOOK.withPreset("mono"));
            editor.open();
            editor.set({motion: "neon"});
            applied.length = 0;
            editor.close(false);
            const reverted = applied.slice();
            const stayed = LOOK.load().motion;
            editor.open();
            editor.set({motion: "aurora"});
            editor.close(true);
            console.log(JSON.stringify({reverted, stayed, saved: LOOK.load().motion}));
        """, sources=("look",))

        assert found["reverted"] == ["none/"], "back to Mono, which has no animation"
        assert found["stayed"] == "none"
        assert found["saved"] == "aurora"

    def test_a_preset_keeps_the_title_somebody_typed(self):
        found = run(EDITOR + """
            editor.open();
            editor.set({title: "Studio"});
            const card = editor.dialog.querySelectorAll(".forge-assistant-look-preset")[1];
            card.handlers.click.forEach((fn) => fn());
            console.log(JSON.stringify({motion: editor.draft.motion, title: editor.draft.title}));
        """, sources=("look",))

        assert found == {"motion": "neon", "title": "Studio"}

    def test_escape_cancels_the_edit(self):
        found = run(EDITOR + """
            editor.open();
            editor.set({motion: "ooze"});
            let prevented = false;
            editor.dialog.handlers.cancel.forEach((fn) => fn({preventDefault() { prevented = true; }}));
            console.log(JSON.stringify({prevented, saved: LOOK.load().motion,
                                        last: applied[applied.length - 1]}));
        """, sources=("look",))

        assert found["prevented"] is True, "taken over only to put the look back first"
        assert found["saved"] == "none"
        assert found["last"] == "none/"


class TestTheShellsSide:
    def test_the_menu_offers_it_only_when_the_module_is_there(self):
        found = run("""
            const shell = Object.create(NS.Shell.prototype);
            shell.state = {freeFloat: false};
            shell.host = {listUtilities: () => []};
            // The menu's New thread item reads the selection.
            shell.store = {snapshot: () => ({selection: {character: "", thread: ""}})};
            const labels = () => shell.utilityItems().map((item) => item.textContent);
            const with_ = labels();
            const look = NS.look;
            delete NS.look;
            const without = labels();
            NS.look = look;
            console.log(JSON.stringify({with_, without}));
        """, sources=("shell", "look"))

        assert "Customize…" in found["with_"]
        assert "Customize…" not in found["without"]

    def test_customizing_puts_the_panel_away_first(self):
        """The launcher is the thing being customized, and it takes each change
        live behind the dialog -- it has to be the thing on screen."""
        found = run("""
            const shell = Object.create(NS.Shell.prototype);
            shell.state = {freeFloat: false, panelOpen: true};
            shell.host = {listUtilities: () => []};
            shell.store = {snapshot: () => ({selection: {character: "", thread: ""}})};
            const order = [];
            shell.closeMenu = () => order.push("menu");
            shell.close = () => order.push("panel");
            NS.look.editor = () => ({open() { order.push("editor"); }});
            const item = shell.utilityItems().find((i) => i.textContent === "Customize\\u2026");
            item.handlers.click.forEach((fn) => fn());
            console.log(JSON.stringify({order}));
        """, sources=("shell", "look"))

        assert found["order"] == ["menu", "panel", "editor"]

    def test_escape_in_the_dialog_does_not_leave_focus_mode(self):
        found = run("""
            const shell = Object.create(NS.Shell.prototype);
            let toggled = 0;
            shell.toggleFocus = () => { toggled += 1; };
            shell.lookEditor = {isOpen: () => true};
            shell.nodes = {root: {contains: () => true}, menu: {hidden: true}};
            shell.focus = {isActive: () => true};
            NS.hostInDialog = () => true;
            NS.escapeOrder = () => "exit-focus";
            let stopped = false;
            shell.documentKey({key: "Escape", preventDefault() {}, stopPropagation() { stopped = true; }});
            console.log(JSON.stringify({toggled, stopped}));
        """, sources=("shell", "look"))

        assert found == {"toggled": 0, "stopped": False}

    def test_the_launchers_name_is_the_title_drawn_on_it(self):
        found = run("""
            const shell = Object.create(NS.Shell.prototype);
            shell.settings = {label: "Forge Assistant", appearance: "text"};
            shell.place = () => {};
            const launcher = document.createElement("button");
            launcher.querySelector = () => null;
            launcher.querySelectorAll = () => [];
            launcher.insertBefore = (c) => c;
            shell.nodes = {launcher, launcherLabel: document.createElement("span"),
                           launcherIcon: document.createElement("span")};
            shell.applyLook(NS.look.sanitize({title: "Studio"}));
            console.log(JSON.stringify({aria: launcher["aria-label"], title: launcher.title}));
        """, sources=("shell", "look"))

        assert found == {"aria": "Studio", "title": "Studio"}
