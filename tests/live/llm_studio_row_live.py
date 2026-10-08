"""LLM Studio's action row on a real Gradio 4.40 page, driven by a real browser.

The node harness in ``tests/test_llm_studio_js.py`` runs the script against
stand-in bubbles, and the first build of the row passed every one of them while
doing nothing on the user's page: Gradio 4.40 draws a message as
``div.message > button[data-testid]``, and a tap that lands inside a button is a
tap the script refused. This script builds the page the panel builds -- the
Chatbot with its id and class, the hidden nomination box, the hidden buttons --
with the real ``javascript/llm_studio.js`` and ``style.css`` and Forge's
``onUiLoaded``/``onAfterUiUpdate`` shape stood in for, opens it in Chromium and
taps.

It needs Gradio 4.40.0 (with the pins Mini Paint NEO's ``tests/requirements.txt``
names) and Playwright, which the suite's own interpreter does not have, so
``tests/test_llm_studio_live.py`` runs it in the interpreter ``MC_GRADIO_PYTHON``
names and skips without one. Run by hand::

    <venv>/bin/python tests/live/llm_studio_row_live.py

It prints one JSON object and exits 0 only when every check held.
"""
from __future__ import annotations

import glob
import json
import os
import socket
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# What the panel writes first in every message (mc_llm_chat_panel._meta), and
# the page the shell wraps the panel in (mc_llm_studio: #mc-llm-studio).
def meta(role: str, key: str, versions: int = 1, active: int = 0) -> str:
    return (f'<span class="mc-llm-meta" data-mc-role="{role}" data-mc-versions="{versions}" '
            f'data-mc-active="{active}" data-mc-key="{key}" hidden></span>')


ROWS = [
    [meta("user", "t1:u0") + "\n\nWhat colour is the sky?",
     meta("assistant", "t1:a0") + "\n\nBlue, on a clear day."],
    [meta("user", "t1:u1") + "\n\nAnd at night?",
     meta("assistant", "t1:a1", versions=3, active=2) + "\n\nBlack, with stars."],
]

NOMINATED = ["regenerate", "edit", "continue", "resend", "branch", "delete",
             "delete-from", "back", "forward", "drop"]

# Forge's script.js, as far as this extension leans on it: gradioApp(), and the
# two callback lists, run from a MutationObserver once the app has rendered.
HEAD = """
<script>
window.gradioApp = function () {
    const app = document.getElementsByTagName("gradio-app")[0];
    return app && app.shadowRoot ? app.shadowRoot : document;
};
const __uiLoaded = [], __afterUpdate = [];
let __loaded = false, __queued = false;
window.onUiLoaded = (fn) => __uiLoaded.push(fn);
window.onAfterUiUpdate = (fn) => __afterUpdate.push(fn);
window.onUiUpdate = () => {};
new MutationObserver(() => {
    if (__queued) return;
    __queued = true;
    requestAnimationFrame(() => {
        __queued = false;
        if (!__loaded && document.querySelector("#mc-llm-chat-transcript")) {
            __loaded = true;
            __uiLoaded.forEach((fn) => { try { fn(); } catch (e) { console.error(e); } });
        }
        if (__loaded) __afterUpdate.forEach((fn) => { try { fn(); } catch (e) { console.error(e); } });
    });
}).observe(document.documentElement, {childList: true, subtree: true, attributes: true, characterData: true});
</script>
<script>
%(script)s
</script>
"""


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def build():
    import gradio as gr

    script = (ROOT / "javascript" / "llm_studio.js").read_text(encoding="utf-8")
    css = (ROOT / "style.css").read_text(encoding="utf-8")
    with gr.Blocks(head=HEAD % {"script": script}, css=css, analytics_enabled=False) as demo:
        with gr.Column(elem_id="mc-llm-studio", elem_classes=["mc-llm-studio"]):
            # The panel turns Gradio's own autoscroll off where the build takes
            # the keyword (mc_llm_chat_panel); 4.40.0 does not.
            import inspect
            extra = ({"autoscroll": False}
                     if "autoscroll" in inspect.signature(gr.Chatbot.__init__).parameters else {})
            gr.Chatbot(label=None, show_label=False, show_copy_button=False, render_markdown=True,
                       value=ROWS, elem_id="mc-llm-chat-transcript",
                       elem_classes=["mc-llm-transcript"], **extra)
            action_at = gr.Textbox(value="", visible=False, container=False,
                                   elem_id="mc-llm-chat-action-at")
            pressed = gr.Textbox(value="", label="pressed", elem_id="pressed", interactive=False)
            for name in NOMINATED:
                button = gr.Button(name, visible=False, elem_id=f"mc-llm-chat-{name}-now")
                button.click(fn=(lambda at, name=name: f"{name}:{at}"), inputs=[action_at],
                             outputs=[pressed], queue=False)
    return demo


def launch_browser(playwright):
    try:
        return playwright.chromium.launch()
    except Exception:
        pass
    base = os.environ.get("PLAYWRIGHT_BROWSERS_PATH") or "/opt/pw-browsers"
    for pattern in ("chromium-*/chrome-linux/chrome", "chromium-*/chrome-linux64/chrome",
                    "chromium_headless_shell-*/chrome-linux/headless_shell"):
        for executable in sorted(glob.glob(str(Path(base) / pattern)), reverse=True):
            try:
                return playwright.chromium.launch(executable_path=executable)
            except Exception:
                continue
    raise RuntimeError("no Chromium to launch")


def main() -> int:
    from playwright.sync_api import sync_playwright

    demo = build()
    port = free_port()
    demo.launch(server_name="127.0.0.1", server_port=port, prevent_thread_lock=True,
                quiet=True, show_api=False, share=False)
    url = f"http://127.0.0.1:{port}/"
    found: dict = {}
    errors: list = []
    try:
        with sync_playwright() as playwright:
            browser = launch_browser(playwright)
            context = browser.new_context(viewport={"width": 1200, "height": 900})
            try:
                context.grant_permissions(["clipboard-read", "clipboard-write"], origin=url)
            except Exception:
                pass
            page = context.new_page()
            page.on("pageerror", lambda error: errors.append(f"pageerror: {error}"))
            # Script errors only: a font or a CDN the sandbox cannot reach is
            # the sandbox's, not the row's.
            page.on("console", lambda message: errors.append(f"console.{message.type}: {message.text}")
                    if message.type == "error" and "Failed to load resource" not in message.text else None)
            page.goto(url)
            page.wait_for_selector('#mc-llm-chat-transcript button[data-testid="bot"]', timeout=30000)
            # The script wires after the host's update; give it its frame.
            page.wait_for_function(
                'document.querySelectorAll("#mc-llm-chat-transcript .mc-llm-message-actions").length >= 1',
                timeout=15000)

            found["shape"] = page.evaluate("""() => {
                const bots = [...document.querySelectorAll('#mc-llm-chat-transcript button[data-testid="bot"]')];
                const users = [...document.querySelectorAll('#mc-llm-chat-transcript button[data-testid="user"]')];
                const rows = [...document.querySelectorAll('#mc-llm-chat-transcript .mc-llm-message-actions')];
                return {
                    bots: bots.length, users: users.length,
                    parent_is_message: bots.every((b) => b.parentElement.classList.contains("message")),
                    rows: rows.length,
                    rows_inside_buttons: document.querySelectorAll('#mc-llm-chat-transcript button[data-testid] .mc-llm-message-actions').length,
                    rows_in_message: rows.filter((r) => r.parentElement.classList.contains("message")).length,
                    rows_hidden: rows.map((r) => getComputedStyle(r).display === "none"),
                    markers: [...document.querySelectorAll('#mc-llm-chat-transcript .mc-llm-meta')].map((m) => ({
                        key: m.getAttribute("data-mc-key"), versions: m.getAttribute("data-mc-versions"),
                        active: m.getAttribute("data-mc-active"), role: m.getAttribute("data-mc-role"),
                        shown: getComputedStyle(m).display !== "none",
                    })),
                    stamped: [...document.querySelectorAll('#mc-llm-chat-transcript .message')].map((m) => m.dataset.mcLlmActions || null),
                    actions: rows.map((r) => [...r.querySelectorAll("button")].map((b) => b.getAttribute("data-action"))),
                    pager: (document.querySelector('#mc-llm-chat-transcript .mc-llm-message-actions-pager') || {}).textContent || null,
                };
            }""")

            # The tap: on the words of the last reply, which sit inside Gradio's button.
            # The words, not the paragraph that holds the (hidden) marker.
            page.click('#mc-llm-chat-transcript button[data-testid="bot"] >> nth=1 >> text=Black, with stars.')
            page.wait_for_function(
                '[...document.querySelectorAll("#mc-llm-chat-transcript .mc-llm-message-actions")].some((r) => getComputedStyle(r).display !== "none")',
                timeout=5000)
            found["tapped"] = page.evaluate("""() => {
                const rows = [...document.querySelectorAll('#mc-llm-chat-transcript .mc-llm-message-actions')];
                const open = rows.map((r) => getComputedStyle(r).display !== "none");
                const bubble = rows[open.indexOf(true)].parentElement;
                const box = bubble.getBoundingClientRect();
                const row = rows[open.indexOf(true)].getBoundingClientRect();
                return {open, revealed: bubble.classList.contains("mc-llm-revealed"),
                        expanded: bubble.getAttribute("aria-expanded"),
                        row_inside_bubble: row.top >= box.top - 1 && row.bottom <= box.bottom + 1,
                        row_height: row.height,
                        buttons_visible: [...rows[open.indexOf(true)].querySelectorAll("button")]
                            .filter((b) => b.getBoundingClientRect().width > 0).length};
            }""")

            # Press Regenerate on it: the server must hear "assistant:1".
            page.click('#mc-llm-chat-transcript .mc-llm-message-actions:not([hidden]) button[data-action="regenerate"]')
            page.wait_for_function(
                '(document.querySelector("#pressed textarea") || {}).value === "regenerate:assistant:1"',
                timeout=10000)
            found["pressed"] = page.evaluate('document.querySelector("#pressed textarea").value')
            found["closed_after_press"] = page.evaluate(
                '[...document.querySelectorAll("#mc-llm-chat-transcript .mc-llm-message-actions")].every((r) => getComputedStyle(r).display === "none")')

            # A tap on a prompt of yours, then a press elsewhere puts it away.
            page.click('#mc-llm-chat-transcript button[data-testid="user"] >> nth=0 >> text=What colour is the sky?')
            page.wait_for_function(
                '[...document.querySelectorAll("#mc-llm-chat-transcript .mc-llm-message-actions")].some((r) => getComputedStyle(r).display !== "none")',
                timeout=5000)
            found["prompt_actions"] = page.evaluate(
                '[...document.querySelector("#mc-llm-chat-transcript .mc-llm-message-actions:not([hidden])").querySelectorAll("button")].map((b) => b.getAttribute("data-action"))')
            page.mouse.click(1100, 850)
            page.wait_for_function(
                '[...document.querySelectorAll("#mc-llm-chat-transcript .mc-llm-message-actions")].every((r) => getComputedStyle(r).display === "none")',
                timeout=5000)
            found["dismissed"] = True

            # The keyboard: focus the host's button and press Enter -- once open, not open-and-shut.
            page.focus('#mc-llm-chat-transcript button[data-testid="bot"] >> nth=0')
            page.keyboard.press("Enter")
            page.wait_for_timeout(300)
            found["keyboard"] = page.evaluate(
                '[...document.querySelectorAll("#mc-llm-chat-transcript .mc-llm-message-actions")].map((r) => getComputedStyle(r).display !== "none")')
            page.keyboard.press("Escape")
            page.wait_for_timeout(200)
            found["escape"] = page.evaluate(
                '[...document.querySelectorAll("#mc-llm-chat-transcript .mc-llm-message-actions")].every((r) => getComputedStyle(r).display === "none")')
            if not found["escape"]:
                page.mouse.click(1100, 850)   # so the rest can still be checked
                page.wait_for_timeout(200)

            # Copy, on the first reply: the words alone.
            page.click('#mc-llm-chat-transcript button[data-testid="bot"] >> nth=0 >> text=Blue, on a clear day.')
            page.wait_for_function(
                '[...document.querySelectorAll("#mc-llm-chat-transcript .mc-llm-message-actions")].some((r) => getComputedStyle(r).display !== "none")',
                timeout=5000)
            page.click('#mc-llm-chat-transcript .mc-llm-message-actions:not([hidden]) button[data-action="copy"]')
            page.wait_for_timeout(300)
            try:
                found["clipboard"] = page.evaluate("navigator.clipboard.readText()")
            except Exception as exc:  # a headless build without a clipboard
                found["clipboard"] = f"unavailable: {exc}"
            found["copy_glyph"] = page.evaluate(
                'document.querySelector("#mc-llm-chat-transcript .mc-llm-message-actions:not([hidden]) button[data-action=copy]").textContent')
            browser.close()
    finally:
        demo.close()

    checks = {
        "two replies and two prompts, each a button inside div.message":
            found["shape"]["bots"] == 2 and found["shape"]["users"] == 2 and found["shape"]["parent_is_message"],
        "the marker survives Gradio's markdown with its attributes, hidden":
            [m["key"] for m in found["shape"]["markers"]] == ["t1:u0", "t1:a0", "t1:u1", "t1:a1"]
            and found["shape"]["markers"][3]["versions"] == "3" and found["shape"]["markers"][3]["active"] == "2"
            and not any(m["shown"] for m in found["shape"]["markers"]),
        "one row per message, beside the host's button, none inside it, all hidden at rest":
            found["shape"]["rows"] == 4 and found["shape"]["rows_in_message"] == 4
            and found["shape"]["rows_inside_buttons"] == 0 and all(found["shape"]["rows_hidden"]),
        "the versioned reply has its pager":
            found["shape"]["pager"] == "3/3",
        "a tap on the words opens that message's row, inside the bubble, with visible buttons":
            found["tapped"]["open"] == [False, False, False, True] and found["tapped"]["revealed"]
            and found["tapped"]["expanded"] == "true" and found["tapped"]["row_inside_bubble"]
            and found["tapped"]["row_height"] > 20 and found["tapped"]["buttons_visible"] >= 9,
        "Regenerate names the message to the server and the row goes away":
            found["pressed"] == "regenerate:assistant:1" and found["closed_after_press"],
        "a prompt's row has Send again and no Regenerate":
            "resend" in found["prompt_actions"] and "regenerate" not in found["prompt_actions"],
        "a press elsewhere puts it away": found["dismissed"],
        # Rows in DOM order: your question, its reply, your question, its reply.
        "Enter on the host's button opens once": found["keyboard"] == [False, True, False, False],
        "Escape closes": found["escape"],
        "Copy puts the words on the clipboard":
            found["clipboard"] in ("Blue, on a clear day.",) or str(found["clipboard"]).startswith("unavailable"),
        "no page errors": not errors,
    }
    found["checks"] = checks
    found["errors"] = errors
    print(json.dumps(found, indent=1, default=str))
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
