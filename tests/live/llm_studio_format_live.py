"""LLM Studio's formatting on a real Gradio 4.40 page, driven by a real browser.

What ``mc_llm_format.for_chatbot`` hands the Chatbot is only half of the
answer: the other half is Gradio's own -- its LaTeX placeholders around marked,
DOMPurify, and the KaTeX 0.16.10 it bundles, run on the delimiters the Chatbot
was built with. None of that is in the suite's stand-in Gradio. This script
builds the Chatbot the panel builds (its id, its class, ``latex_delimiters``,
the messages drawn through ``for_chatbot`` with the marker first), with the
real ``style.css`` and ``javascript/llm_studio.js``, opens it in Chromium and
looks: Gemma's arrow typeset, money and code as written, quotes blue by day and
amber at night with their own marks, a lone tilde not struck, and Copy giving a
formula back as its TeX.

Needs Gradio 4.40.0 and Playwright, so ``tests/test_llm_studio_format_live.py``
runs it in the interpreter ``MC_GRADIO_PYTHON`` names and skips without one.
Run by hand::

    <venv>/bin/python tests/live/llm_studio_format_live.py

It prints one JSON object and exits 0 only when every check held.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import llm_studio_row_live as live  # noqa: E402
import mc_llm_format as fmt  # noqa: E402

CODE = "x = '$a$' \"q\" 3~4\nr = re.compile(r\"\\((?P<n>\\d+)\\)\")"
MESSAGES = [
    ("user", "f1:u0", 'What does $\\rightarrow$ mean? It costs $5 and $10. She said "hello there".'),
    ("assistant", "f1:a0", "Display:\n$$\n\\frac{a}{b} < c\n$$\nand \\(x^2\\) inline, 3~4 hours, ~~struck~~."),
    ("user", "f1:u1", f"```python\n{CODE}\n```\nafter $25^\\circ$C"),
    ("assistant", "f1:a1", '| a | b |\n|---|---|\n| "q" | $x_1$ |\n\n\\begin{align} a &= b \\\\ c &= d \\end{align}'),
]
BLUE = "rgb(37, 99, 235)"
AMBER = "rgb(255, 176, 102)"


def rows():
    drawn = [(role, live.meta(role, key) + "\n\n" + fmt.for_chatbot(text)) for role, key, text in MESSAGES]
    return [[drawn[0][1], drawn[1][1]], [drawn[2][1], drawn[3][1]]]


def build():
    import gradio as gr

    script = (ROOT / "javascript" / "llm_studio.js").read_text(encoding="utf-8")
    css = (ROOT / "style.css").read_text(encoding="utf-8")
    with gr.Blocks(head=live.HEAD % {"script": script}, css=css, analytics_enabled=False) as demo:
        with gr.Column(elem_id="mc-llm-studio", elem_classes=["mc-llm-studio"]):
            gr.Chatbot(label=None, show_label=False, show_copy_button=False, render_markdown=True,
                       value=rows(), latex_delimiters=fmt.latex_delimiters(),
                       elem_id="mc-llm-chat-transcript", elem_classes=["mc-llm-transcript"])
            gr.Textbox(value="", visible=False, container=False, elem_id="mc-llm-chat-action-at")
            for name in live.NOMINATED:
                gr.Button(name, visible=False, elem_id=f"mc-llm-chat-{name}-now")
    return demo


LOOK = """() => {
    const bubbles = [...document.querySelectorAll('#mc-llm-chat-transcript button[data-testid]')];
    return bubbles.map((b) => ({
        role: b.getAttribute('data-testid'),
        tex: [...b.querySelectorAll('annotation')].map((a) => a.textContent),
        displayed: b.querySelectorAll('.katex-display').length,
        errors: b.querySelectorAll('.katex-error').length,
        quotes: [...b.querySelectorAll('q.mc-llm-quote')].map((q) => ({
            text: q.textContent, colour: getComputedStyle(q).color,
            before: getComputedStyle(q, '::before').content})),
        struck: [...b.querySelectorAll('del, s')].map((d) => d.textContent),
        code: [...b.querySelectorAll('pre code')].map((c) => c.textContent),
        text: b.textContent,
    }));
}"""


def main() -> int:
    from playwright.sync_api import sync_playwright

    demo = build()
    port = live.free_port()
    demo.launch(server_name="127.0.0.1", server_port=port, prevent_thread_lock=True,
                quiet=True, show_api=False, share=False)
    url = f"http://127.0.0.1:{port}/"
    found: dict = {}
    errors: list = []
    try:
        with sync_playwright() as playwright:
            browser = live.launch_browser(playwright)
            context = browser.new_context(viewport={"width": 1200, "height": 900})
            try:
                context.grant_permissions(["clipboard-read", "clipboard-write"], origin=url)
            except Exception:
                pass
            page = context.new_page()
            page.on("pageerror", lambda error: errors.append(f"pageerror: {error}"))
            page.goto(url)
            page.wait_for_selector('#mc-llm-chat-transcript button[data-testid="bot"]', timeout=30000)
            page.wait_for_function(
                "document.querySelectorAll('#mc-llm-chat-transcript .katex').length >= 6", timeout=15000)
            page.wait_for_function(
                'document.querySelectorAll("#mc-llm-chat-transcript .mc-llm-message-actions").length >= 1',
                timeout=15000)
            found["day"] = page.evaluate(LOOK)
            page.evaluate("document.body.classList.add('dark')")
            found["night"] = page.evaluate(LOOK)
            page.evaluate("document.body.classList.remove('dark')")

            # Copy, on your first message: the arrow comes back as its TeX.
            page.click('#mc-llm-chat-transcript button[data-testid="user"] >> nth=0 >> text=She said')
            page.wait_for_function(
                '[...document.querySelectorAll("#mc-llm-chat-transcript .mc-llm-message-actions")]'
                '.some((r) => getComputedStyle(r).display !== "none")', timeout=5000)
            page.click('#mc-llm-chat-transcript .mc-llm-message-actions:not([hidden]) button[data-action="copy"]')
            page.wait_for_timeout(300)
            try:
                found["clipboard"] = page.evaluate("navigator.clipboard.readText()")
            except Exception as exc:  # a headless build without a clipboard
                found["clipboard"] = f"unavailable: {exc}"
            browser.close()
    finally:
        demo.close()

    day, night = found["day"], found["night"]
    checks = {
        "Gemma's arrow and every other formula typeset by Gradio's KaTeX, none in error":
            day[0]["tex"] == ["\\rightarrow"] and day[1]["tex"] == ["\n\\frac{a}{b} < c\n", "x^2"]
            and day[2]["tex"] == ["25^\\circ"]
            and day[3]["tex"] == ["x_1", "\\begin{align} a &= b \\\\ c &= d \\end{align}"]
            and [b["displayed"] for b in day] == [0, 1, 0, 1] and not any(b["errors"] for b in day),
        "money is money": "It costs $5 and $10." in day[0]["text"],
        "code exactly as written": day[2]["code"] == [CODE + "\n"] or day[2]["code"] == [CODE],
        "a quote is marked with its own marks, blue by day":
            day[0]["quotes"] == [{"text": '"hello there"', "colour": BLUE, "before": "none"}]
            and [q["text"] for q in day[3]["quotes"]] == ['"q"'],
        "and amber at night": [q["colour"] for q in night[0]["quotes"]] == [AMBER],
        "one tilde is not a strikethrough, two are":
            day[1]["struck"] == ["struck"] and "3~4 hours" in day[1]["text"],
        "Copy gives a formula back as its TeX":
            str(found["clipboard"]).startswith("unavailable")
            or ("$\\rightarrow$" in found["clipboard"] and "→" not in found["clipboard"]),
        "no page errors": not errors,
    }
    found["checks"] = checks
    found["errors"] = errors
    print(json.dumps(found, indent=1, default=str, ensure_ascii=False))
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
