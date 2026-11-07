"""A message's math, quotes and code: one reading, two surfaces.

Asked for: "Sometimes gemma4 responds with syntax like "$\\rightarrow$", i want
for all of the syntax formats to be handled correctly. Also, when text is in
quotes like "this" i want it to be slightly more highlight."

LLM Studio's transcript is drawn by Gradio's Chatbot from what
``mc_llm_format.for_chatbot`` hands it; the Forge Assistant's panel draws its
own (``javascript/forge_assistant_markdown.js``). The two carry the same rules
in two languages, so they are held to one table of cases
(``tests/fixtures/chat_format_cases.json``) and to each other on generated
messages; then what LLM Studio is given is checked, and that the thread keeps
exactly what was written.
"""

from __future__ import annotations

import json
import random
import shutil
import subprocess
import unicodedata
from pathlib import Path

import pytest

import mc_llm_format as fmt

ROOT = Path(__file__).resolve().parent.parent
CASES = json.loads((ROOT / "tests" / "fixtures" / "chat_format_cases.json").read_text(encoding="utf-8"))
RENDERER = ROOT / "javascript" / "forge_assistant_markdown.js"

needs_node = pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")

DESCRIBE_IN_NODE = """
const fs = require("fs");
globalThis.window = globalThis;
eval(fs.readFileSync(process.argv[1], "utf8"));
const texts = JSON.parse(fs.readFileSync(0, "utf8"));
console.log(JSON.stringify(texts.map((text) => globalThis.forgeAssistant.describeMessage(text))));
"""


def describe_in_node(texts: list) -> list:
    """The panel's reading of each text, from the real script under node."""
    done = subprocess.run(["node", "-e", DESCRIBE_IN_NODE, str(RENDERER)], input=json.dumps(texts),
                          capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout)


# --------------------------------------------------------------------------- #
# The shared table
# --------------------------------------------------------------------------- #


class TestTheSharedTable:
    @pytest.mark.parametrize("case", CASES, ids=[case["name"] for case in CASES])
    def test_llm_studio_reads_each_case_as_written_down(self, case):
        assert fmt.describe(case["text"]) == {"math": case["math"], "quotes": case["quotes"]}

    @needs_node
    def test_the_panel_reads_every_case_the_same_way(self):
        found = describe_in_node([case["text"] for case in CASES])

        wrong = [(case["name"], got) for case, got in zip(CASES, found)
                 if got != {"math": case["math"], "quotes": case["quotes"]}]
        assert not wrong

    def test_the_table_covers_what_was_asked_about(self):
        """Gemma's arrow, money that must stay money, and a quotation."""
        names = {case["name"] for case in CASES}
        assert {"gemma_arrow", "money_pairs", "money_ranges", "quotes"} <= names
        arrow = next(case for case in CASES if case["name"] == "gemma_arrow")
        assert arrow["text"] == "Then $\\rightarrow$ go"

    @needs_node
    def test_the_two_readings_agree_on_generated_messages(self):
        """Seeded soups of every token the rules turn on, run through both."""
        tokens = ["$", "$$", "\\(", "\\)", "\\[", "\\]", "`", "``", "```", "~~~", '"', "“",
                  "”", "„", "«", "»", "|", "\n", "\n\n", "x", "y2", " ", "  ",
                  "\\", "~", "~~", "<b>", "</b>", '<a href="q">', "](", ")", "(", "[", "]",
                  "\\begin{align}", "\\end{align}", "\\begin{pmatrix}", "\\end{pmatrix}", "    ",
                  "- ", "1. ", "5", "\\$", '\\"', "&", "\t", "word", ".", ",", "「", "」",
                  "*", "_", "<https://a.b>", "é", "日"]
        chance = random.Random(20261105)
        texts = ["".join(chance.choice(tokens) for _ in range(chance.randint(1, 40)))
                 for _ in range(1500)]

        found = describe_in_node(texts)

        wrong = [text for text, python, script in zip(texts, map(fmt.describe, texts), found)
                 if python != script]
        assert not wrong, wrong[:3]


# --------------------------------------------------------------------------- #
# What LLM Studio's Chatbot is given
# --------------------------------------------------------------------------- #


class TestWhatTheChatbotIsGiven:
    def test_gemmas_arrow_reaches_the_chatbot_between_its_own_delimiters(self):
        left, right = fmt.INLINE_MATH

        assert fmt.for_chatbot("Then $\\rightarrow$ go") == f"Then {left}\\rightarrow{right} go"

    def test_displayed_math_takes_the_display_pair_whichever_way_it_was_written(self):
        left, right = fmt.DISPLAY_MATH

        assert fmt.for_chatbot("$$x$$") == f"{left}x{right}"
        assert fmt.for_chatbot("\\[x\\]") == f"{left}x{right}"
        assert fmt.for_chatbot("\\begin{equation}x\\end{equation}") \
            == f"{left}\\begin{{equation}}x\\end{{equation}}{right}"

    def test_inline_math_takes_the_inline_pair_whichever_way_it_was_written(self):
        left, right = fmt.INLINE_MATH

        assert fmt.for_chatbot("\\(x\\)") == f"{left}x{right}"
        assert fmt.for_chatbot("$x$") == f"{left}x{right}"

    def test_math_is_escaped_for_html_because_gradio_puts_it_back_as_html(self):
        """Gradio lifts each delimited piece out before marked and puts it back
        into the HTML exactly as it found it: an unescaped `<b` would open a
        tag, and the formula would lose the rest of itself to the sanitiser."""
        left, right = fmt.INLINE_MATH

        assert fmt.for_chatbot("$a<b \\& c>d$") == f"{left}a&lt;b \\&amp; c&gt;d{right}"

    def test_money_and_code_reach_it_exactly_as_written(self):
        for text in ("It costs $5 and $10 total", "From $1,000-$2,000",
                     "`echo $HOME` and `$x$`", "```bash\necho $$ \"$x\" 3~4\n```",
                     "Para\n\n    $code$ \"q\" 1~2\n"):
            assert fmt.for_chatbot(text) == text

    def test_a_quote_is_marked_with_its_own_marks_inside(self):
        assert fmt.for_chatbot('She said "hi" and left') \
            == 'She said <span class="mc-llm-quote">"hi"</span> and left'
        assert fmt.for_chatbot("“Curly”") \
            == '<span class="mc-llm-quote">“Curly”</span>'

    def test_a_quotation_is_a_span_never_a_q_element(self):
        """A Gecko browser's plain-text copy writes a quote mark at each end of
        a <q> on top of the message's own, so a selection copied from LibreWolf
        turned "Hello" into ""Hello"" (2026-11-06; checked in a Gecko build)."""
        assert fmt.for_chatbot('Character stepped in the room "Hello" and waves.') == (
            'Character stepped in the room <span class="mc-llm-quote">"Hello"</span> and waves.')
        for case in CASES:
            assert "<q" not in fmt.for_chatbot(case["text"]), case["text"]

    def test_a_lone_tilde_is_escaped_and_a_double_one_is_still_a_strikethrough(self):
        """Gradio 4.40's marked strikes out what lies between two single tildes."""
        assert fmt.for_chatbot("3~4 hours and 5~6 days ~~gone~~") \
            == "3\\~4 hours and 5\\~6 days ~~gone~~"

    def test_a_message_with_nothing_to_format_is_unchanged(self):
        text = "# Title\n\n*   **Bold:** one\n*   two\n\n| a | b |\n|---|---|\n| 1 | 2 |"
        assert fmt.for_chatbot(text) == text

    def test_carriage_returns_are_lines(self):
        assert fmt.for_chatbot("a\r\nb\rc") == "a\nb\nc"

    def test_nothing_is_nothing(self):
        assert fmt.for_chatbot("") == ""
        assert fmt.for_chatbot(None) == ""

    def test_the_delimiters_are_invisible_and_told_apart(self):
        """Default-ignorable format characters, so a browser draws nothing
        where one is left as text; and neither left one is the start of the
        other, which KaTeX's auto-render needs to tell them apart."""
        for character in "".join(fmt.INLINE_MATH + fmt.DISPLAY_MATH):
            assert unicodedata.category(character) == "Cf"
        assert not fmt.INLINE_MATH[0].startswith(fmt.DISPLAY_MATH[0])
        assert not fmt.DISPLAY_MATH[0].startswith(fmt.INLINE_MATH[0])
        assert fmt.latex_delimiters() == [
            {"left": fmt.INLINE_MATH[0], "right": fmt.INLINE_MATH[1], "display": False},
            {"left": fmt.DISPLAY_MATH[0], "right": fmt.DISPLAY_MATH[1], "display": True},
        ]

    def test_no_dollar_is_a_delimiter_for_the_chatbot(self):
        """Every `$` on the page would be KaTeX's to take, money included."""
        for pair in fmt.latex_delimiters():
            assert "$" not in pair["left"] and "$" not in pair["right"]


# --------------------------------------------------------------------------- #
# The panel's wiring
# --------------------------------------------------------------------------- #


class _Message:
    def __init__(self, role, text):
        self.role = role
        self.text = text
        self.versions = [text]
        self.active = 0
        self.attached = False
        self.image_path = ""
        self.image = ""
        self.image_name = ""


class _Conversation:
    identifier = "thread-1"
    character = "Ada"
    title = "Arrows"

    def __init__(self, messages):
        self.messages = messages


class TestTheTranscript:
    @pytest.fixture(autouse=True)
    def store(self, tmp_path, monkeypatch, host):
        """Building the panel reads and writes the LLM data directory: here, a
        temporary one, never the checkout's (tests/conftest.py's guard)."""
        import mc_llm_paths

        monkeypatch.setattr(mc_llm_paths, "data_root", lambda: tmp_path)
        yield tmp_path

    def test_every_message_is_drawn_through_the_reading_and_saved_as_written(self):
        import mc_llm_chat_panel as panel
        from prompt_master.chat.history import ASSISTANT, USER

        words = 'Then $\\rightarrow$, "quoted", 3~4'
        conversation = _Conversation([_Message(USER, words), _Message(ASSISTANT, words)])

        rows, _positions = panel._view(conversation)

        drawn = fmt.for_chatbot(words)
        assert rows[0][0].endswith(drawn) and rows[0][1].endswith(drawn)
        assert all(message.text == words for message in conversation.messages), \
            "the thread keeps what was written"

    def test_the_markers_key_is_taken_from_the_words_as_saved(self):
        """The Voice Box files a render under the key; a key taken from the
        drawn copy would change whenever the drawing does."""
        import hashlib

        import mc_llm_chat_panel as panel
        from prompt_master.chat.history import ASSISTANT

        words = 'A "quote" and $x$'
        conversation = _Conversation([_Message(ASSISTANT, words)])

        marker = panel._meta(conversation, conversation.messages[0])

        digest = hashlib.sha1(f"{ASSISTANT}\n{words}".encode("utf-8")).hexdigest()[:12]
        assert f'data-mc-key="thread-1:{digest}"' in marker

    def test_a_picture_is_put_before_the_words_untouched(self):
        import mc_llm_chat_panel as panel
        from prompt_master.chat.history import USER

        message = _Message(USER, 'look "here"')
        message.attached = True
        message.image = "data:image/png;base64,AAAA"
        message.image_name = 'a "named" picture'

        body = panel._body(message)

        assert body.startswith('<img src="data:image/png;base64,AAAA"')
        assert body.endswith('<span class="mc-llm-quote">"here"</span>')

    def test_the_chatbot_is_built_with_the_two_invisible_pairs_as_its_delimiters(self, monkeypatch):
        import gradio as gr

        import mc_llm_chat_panel as panel

        built = []

        class Recorded(gr.Chatbot):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                built.append(kwargs)

        monkeypatch.setattr(gr, "Chatbot", Recorded)
        panel.build()

        assert len(built) == 1
        assert built[0]["latex_delimiters"] == fmt.latex_delimiters()
        assert built[0]["render_markdown"] is True
