"""What is free, in the header's empty space.

Asked for from use: "In the empty space of our drag handle, i want gray text
italicized ... the free ram from my system, and vram from my two gpu ... tell
me what is free on the chips in real time if possible, but i do not want to
open a new live connection."

So the panel asks with a plain request, one at a time, every MEMORY_EVERY while
it is open and the page is visible, and asks for nothing otherwise. These hold
that: the figures, the cadence, and the two moments it must stop.
"""

from __future__ import annotations

import re

from test_assistant_js import SHELL, run


CLOCK = """
const timers = [];
globalThis.setTimeout = (fn, ms) => { timers.push({fn, ms, live: true}); return timers.length; };
globalThis.clearTimeout = (id) => { if (timers[id - 1]) timers[id - 1].live = false; };
const live = () => timers.filter((timer) => timer.live);
const lapse = () => {
    const due = live();
    due.forEach((timer) => { timer.live = false; timer.fn(); });
    return due.length;
};
const settle = () => new Promise((resolve) => setImmediate(resolve));

const READING = {ok: true, ram: {free: 61.2 * 2 ** 30, total: 96 * 2 ** 30},
                 cards: [{index: 0, name: "5090", full_name: "NVIDIA GeForce RTX 5090",
                          free: 18.4 * 2 ** 30, total: 32 * 2 ** 30},
                         {index: 1, name: "3090", full_name: "NVIDIA GeForce RTX 3090",
                          free: 7.06 * 2 ** 30, total: 24 * 2 ** 30}]};

function memoryShell(answer) {
    const shell = Object.create(NS.Shell.prototype);
    shell.state = {panelOpen: true};
    shell.asked = [];
    shell.store = {request(path, options) {
        shell.asked.push({path, deadline: options && options.deadline});
        return answer();
    }};
    shell.nodes = {reading: element("reading", "SPAN"), grip: element("grip")};
    return shell;
}
"""


def test_the_figures_are_gigabytes_to_one_place_each_held_to_its_name():
    found = run("""
        console.log(JSON.stringify(NS.memoryText({
            ok: true, ram: {free: 61.2 * 2 ** 30, total: 96 * 2 ** 30},
            cards: [{name: "5090", full_name: "NVIDIA GeForce RTX 5090",
                     free: 18.4 * 2 ** 30, total: 32 * 2 ** 30},
                    {name: "RTX A6000", free: 0, total: 48 * 2 ** 30}]})));
    """)

    assert found["text"] == "RAM 61.2 5090 18.4 RTX A6000 0.0"
    assert found["title"] == ("Free: System RAM 61.2 of 96.0 GB; "
                              "NVIDIA GeForce RTX 5090 18.4 of 32.0 GB; RTX A6000 0.0 of 48.0 GB")


def test_a_part_that_could_not_be_read_is_left_out():
    found = run("""
        console.log(JSON.stringify([NS.memoryText({ok: true, ram: null, cards: []}),
                                    NS.memoryText(null),
                                    NS.memoryText({ram: {free: 2 ** 30}, cards: []})]));
    """)

    assert found[0] == {"text": "", "title": ""}
    assert found[1] == {"text": "", "title": ""}
    assert found[2]["text"] == "RAM 1.0"


def test_it_asks_paints_and_asks_again_after_the_answer():
    found = run(CLOCK + """
        const shell = memoryShell(() => Promise.resolve(READING));
        shell.readMemory();
        const before = live().length;
        await settle();
        const painted = shell.nodes.reading.textContent;
        const waiting = live().map((timer) => timer.ms);
        lapse();
        await settle();
        console.log(JSON.stringify({before, painted, waiting, asked: shell.asked,
                                    title: shell.nodes.grip.title}));
    """)

    assert found["before"] == 0, "the next ask is timed from the answer, not from the ask"
    assert found["painted"] == "RAM 61.2 5090 18.4 3090 7.1"
    assert found["waiting"] == [5000]
    assert [ask["path"] for ask in found["asked"]] == ["/memory", "/memory"]
    assert all(ask["deadline"] == 4000 for ask in found["asked"]), "every ask has a deadline"
    assert found["title"].startswith("Free: System RAM 61.2 of 96.0 GB")


def test_a_closed_panel_asks_for_nothing_and_an_answer_on_its_way_is_dropped():
    found = run(CLOCK + """
        let release;
        const shell = memoryShell(() => new Promise((resolve) => { release = resolve; }));
        shell.readMemory();
        shell.state.panelOpen = false;
        shell.stopMemory();
        release(READING);
        await settle();
        const after = {painted: shell.nodes.reading.textContent, timers: live().length};
        shell.readMemory();
        console.log(JSON.stringify(Object.assign(after, {asked: shell.asked.length})));
    """)

    assert found == {"painted": "", "timers": 0, "asked": 1}


def test_a_hidden_page_asks_for_nothing():
    found = run(CLOCK + """
        const shell = memoryShell(() => Promise.resolve(READING));
        document.hidden = true;
        shell.readMemory();
        await settle();
        console.log(JSON.stringify({asked: shell.asked.length, timers: live().length}));
    """)

    assert found == {"asked": 0, "timers": 0}


def test_a_failed_ask_keeps_the_last_figures_dimmed_and_tries_again():
    found = run(CLOCK + """
        let fail = false;
        const shell = memoryShell(() => fail ? Promise.reject(new Error("deadline"))
                                              : Promise.resolve(READING));
        shell.readMemory();
        await settle();
        fail = true;
        lapse();
        await settle();
        const stale = {text: shell.nodes.reading.textContent,
                       dim: shell.nodes.reading.classList.contains(
                           "forge-assistant-memory-stale"),
                       again: live().length};
        fail = false;
        lapse();
        await settle();
        stale.fresh = !shell.nodes.reading.classList.contains("forge-assistant-memory-stale");
        console.log(JSON.stringify(stale));
    """)

    assert found == {"text": "RAM 61.2 5090 18.4 3090 7.1",
                     "dim": True, "again": 1, "fresh": True}


def test_opening_starts_it_closing_stops_it_and_a_repair_does_not_double_it():
    """`heal` calls `showOpen` again, and must not start a second loop."""
    found = run(CLOCK + """
        const shell = memoryShell(() => new Promise(() => {}));
        shell.nodes.panel = element("panel");
        shell.nodes.launcher = element("launcher");
        shell.closeMenu = () => {};
        shell._save = () => {};
        shell.placeNow = () => {};
        shell.showOpen(true);
        shell.showOpen(true);
        const open = shell.asked.length;
        shell.showOpen(false);
        console.log(JSON.stringify({open, busy: !!shell._memoryBusy}));
    """)

    assert found == {"open": 1, "busy": False}


def test_the_reading_is_out_of_the_hand_s_way():
    """A press on the figures has to be a press on the grip, or the panel
    could not be dragged by the very space the figures sit in."""
    css = (SHELL.parent.parent / "style.css").read_text(encoding="utf-8")
    rule = css.split(".forge-assistant-memory {", 1)[1].split("}", 1)[0]

    assert "pointer-events: none" in rule
    assert "font-style: italic" in rule
    assert re.search(r"color:\s*var\(--body-text-color-subdued", rule)


def test_the_reading_carries_the_lora_mode_to_the_menu_and_the_tooltip():
    """The ⋯ menu's Warm LoRA / Cold LoRA entries read the server's state off
    the free-memory reading, so they need no request of their own."""
    found = run(CLOCK + """
        const shell = memoryShell(() => Promise.resolve(READING));
        shell.paintMemory(Object.assign({}, READING, {
            lora: {mode: "warm", merged: true, blob: false, originals_bytes: 12.1 * 2 ** 30}}), true);
        const warm = {mode: shell.loraRam.mode, title: shell.nodes.reading.title};
        shell.paintMemory(Object.assign({}, READING, {
            lora: {mode: "cold", merged: false, blob: true, originals_bytes: 0}}), true);
        const cold = {mode: shell.loraRam.mode, title: shell.nodes.reading.title};
        // A stale repaint, and a reading without the block, leave the mode alone.
        shell.paintMemory(Object.assign({}, READING, {lora: {mode: "warm"}}), false);
        shell.paintMemory(READING, true);
        console.log(JSON.stringify({warm, cold, kept: shell.loraRam.mode}));
    """)

    assert found["warm"]["mode"] == "warm"
    assert "LoRA originals 12.1 GB in system RAM (Warm LoRA)" in found["warm"]["title"]
    assert found["cold"]["mode"] == "cold"
    assert "LoRA baked in (Cold LoRA)" in found["cold"]["title"]
    assert "LoRA originals" not in found["cold"]["title"]
    assert found["kept"] == "cold"

