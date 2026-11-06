"""The Forge Assistant's renderer of a message's words, run under node.

``javascript/forge_assistant_markdown.js`` draws what a model writes --
headings, lists, tables, block quotes, rules, code, emphasis, links, quotes and
math -- as HTML the panel inserts, every element and attribute one the file
writes, a formula KaTeX's own drawing with ``trust`` off. Asked for: "i want
for all of the syntax formats to be handled correctly", and quotes "slightly
more highlight". Its reading of where math and quotes are is held to
mc_llm_format.py's in ``tests/test_chat_format.py``; this is what it draws.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from test_assistant_js import run

ROOT = Path(__file__).resolve().parent.parent
KATEX = ROOT / "vendor" / "katex" / "katex.min.js"


def draw(*texts, katex: bool = False) -> list:
    """Each text drawn by the real renderer; with the vendored KaTeX loaded as
    the panel loads it -- the UMD bundle run with a module of its own."""
    loading = f"""
        const fs = await import("fs");
        const holder = {{exports: {{}}}};
        new Function("module", "exports", "define", fs.readFileSync({json.dumps(str(KATEX))}, "utf8"))(
            holder, holder.exports, undefined);
        NS.useKatex(holder.exports);
    """ if katex else ""
    found = run(loading + f"""
        const texts = {json.dumps(list(texts))};
        console.log(JSON.stringify({{out: texts.map((text) => NS.renderMarkdown(text))}}));
    """, sources=("markdown",))
    return found["out"]


def one(text, katex: bool = False) -> str:
    return draw(text, katex=katex)[0]


class TestBlocks:
    def test_headings_by_hashes_and_by_underlines(self):
        out = draw("# One", "### Three ###", "Title\n===", "Sub\n---", "#hashtag")

        assert out[:4] == ["<h1>One</h1>", "<h3>Three</h3>", "<h1>Title</h1>", "<h2>Sub</h2>"]
        assert out[4] == "<p>#hashtag</p>"

    def test_paragraphs_and_the_line_breaks_inside_them(self):
        assert one("one\ntwo\n\nthree") == "<p>one<br>two</p><p>three</p>"

    def test_gemmas_bullets_nest_by_indentation(self):
        """Gemma writes `*   ` and nests by four spaces."""
        out = one("*   **One:** first\n*   Two\n    *   Nested\n    *   Again\n*   Three")

        assert out == ("<ul><li><strong>One:</strong> first</li><li>Two<ul><li>Nested</li>"
                       "<li>Again</li></ul></li><li>Three</li></ul>")

    def test_a_numbered_list_keeps_its_first_number(self):
        assert one("1. a\n2. b") == "<ol><li>a</li><li>b</li></ol>"
        assert one("3) c\n4) d") == '<ol start="3"><li>c</li><li>d</li></ol>'

    def test_a_list_with_space_between_its_items_has_paragraphs(self):
        assert one("- a\n\n- b") == "<ul><li><p>a</p></li><li><p>b</p></li></ul>"

    def test_a_task_list_is_drawn_with_boxes_not_inputs(self):
        out = one("- [ ] todo\n- [x] done")

        assert "<input" not in out
        assert out.count('class="forge-assistant-task"') == 2
        assert "☐</span> todo" in out and "☑</span> done" in out

    def test_a_list_after_a_paragraph_line_starts_without_a_blank_line(self):
        assert one("Steps:\n- a\n- b") == "<p>Steps:</p><ul><li>a</li><li>b</li></ul>"

    def test_a_number_that_is_not_one_does_not_start_a_list_inside_a_paragraph(self):
        assert one("In\n2026. it rained") == "<p>In<br>2026. it rained</p>"

    def test_a_block_quote_holds_its_blocks(self):
        assert one("> said\n> again\n\n> - a\n> - b") == (
            "<blockquote><p>said<br>again</p></blockquote>"
            "<blockquote><ul><li>a</li><li>b</li></ul></blockquote>")

    def test_rules(self):
        assert draw("a\n\n---", "***", "* * *", "___") == [
            "<p>a</p><hr>", "<hr>", "<hr>", "<hr>"]

    def test_a_table_with_its_alignments_and_its_cells_formatted(self):
        out = one("| Name | Left | Mid | Right |\n|---|:--|:-:|--:|\n| **a** | `b` | c \\| d | e |")

        assert out.startswith("<table><thead><tr><th>Name</th>")
        assert '<th class="forge-assistant-align-left">Left</th>' in out
        assert '<th class="forge-assistant-align-center">Mid</th>' in out
        assert '<td class="forge-assistant-align-right">e</td>' in out
        assert "<td><strong>a</strong></td>" in out
        assert '<td class="forge-assistant-align-center">c | d</td>' in out, "an escaped pipe is a pipe"

    def test_a_table_without_the_outer_pipes(self):
        out = one("a | b\n--- | ---\n1 | 2")

        assert out == ("<table><thead><tr><th>a</th><th>b</th></tr></thead>"
                       "<tbody><tr><td>1</td><td>2</td></tr></tbody></table>")

    def test_a_fenced_block_is_shown_exactly_as_written(self):
        out = one('```python\nx = "$a$" **b** <i>\n```')

        assert out == ('<pre class="forge-assistant-code"><code>x = &quot;$a$&quot; **b** '
                       "&lt;i&gt;</code></pre>")

    def test_a_fence_left_open_while_a_reply_streams_runs_to_the_end(self):
        assert one("```\n**not yet**") == '<pre class="forge-assistant-code"><code>**not yet**</code></pre>'

    def test_an_indented_block_after_prose_is_code_and_one_in_a_list_is_not(self):
        out = one("Here:\n\n    $x$ code\n\n- item\n\n    more $y$")

        assert '<pre class="forge-assistant-code"><code>$x$ code</code></pre>' in out
        assert "<li><p>item</p><p>more " in out


class TestInlines:
    def test_emphasis_as_commonmark_pairs_it(self):
        out = draw("***both***", "*a **b** c*", "**a *b* c**", "snake_case_word", "__init__",
                   "2 * 3 * 4", "a*b*c")

        assert out == ["<p><em><strong>both</strong></em></p>",
                       "<p><em>a <strong>b</strong> c</em></p>",
                       "<p><strong>a <em>b</em> c</strong></p>",
                       "<p>snake_case_word</p>",
                       "<p><strong>init</strong></p>",
                       "<p>2 * 3 * 4</p>",
                       "<p>a<em>b</em>c</p>"]

    def test_two_tildes_strike_and_one_does_not(self):
        """Gradio's marked strikes between single tildes; this does not, and
        LLM Studio's text is escaped so that neither does it."""
        assert one("~~gone~~ but 3~4 hours and 5~6 days") \
            == "<p><del>gone</del> but 3~4 hours and 5~6 days</p>"

    def test_code_spans(self):
        assert draw("call `now()`", "``a ` b``", "` padded `") == [
            "<p>call <code>now()</code></p>", "<p><code>a ` b</code></p>",
            "<p><code>padded</code></p>"]

    def test_links_pictures_and_addresses(self):
        out = draw("[go](https://a.b/c \"title\")", "![a cat](https://a.b/cat.png)",
                   "<https://x.y/z>", "Visit https://ex.am/a_(b). Or www.ex.am, please.",
                   "mail <me@ex.am>")

        assert out[0] == ('<p><a href="https://a.b/c" target="_blank" '
                          'rel="noopener noreferrer">go</a></p>')
        assert "<img" not in out[1] and 'href="https://a.b/cat.png"' in out[1] and "a cat" in out[1]
        assert 'href="https://x.y/z"' in out[2]
        assert 'href="https://ex.am/a_(b)"' in out[3] and "</a>." in out[3]
        assert 'href="http://www.ex.am"' in out[3] and "www.ex.am</a>," in out[3]
        assert 'href="mailto:me@ex.am"' in out[4]

    def test_entities_are_their_characters_and_unknown_ones_their_text(self):
        assert one("&rarr; &#8594; &#x2192; &amp; &lt;b&gt; &bogus;") \
            == "<p>→ → → &amp; &lt;b&gt; &amp;bogus;</p>"

    def test_escapes(self):
        assert one("\\*not\\* \\# \\$5 \\_x\\_ \\`y\\`") == "<p>*not* # $5 _x_ `y`</p>"

    def test_an_escaped_bracket_pair_is_displayed_math_as_in_the_rules(self):
        """`\\[ … \\]` is LaTeX's displayed math, and a model writes it for that;
        Markdown's escaped brackets are the price, on both surfaces alike."""
        assert 'data-display="1">\\[x\\]</span>' in one("see \\[x\\]")

    def test_the_inline_tags_a_model_writes_are_drawn_again_and_others_shown(self):
        out = draw("a<br>b<br/>c<br />d", "<sup>2</sup> <kbd>Ctrl</kbd> <u>u</u>",
                   "<b>open", "</i>shut", '<b class="x">bold</b>', "<span>s</span>")

        assert out[0] == "<p>a<br>b<br>c<br>d</p>"
        assert out[1] == "<p><sup>2</sup> <kbd>Ctrl</kbd> <u>u</u></p>"
        assert out[2] == "<p>&lt;b&gt;open</p>", "never closed: its text"
        assert out[3] == "<p>&lt;/i&gt;shut</p>"
        assert "<b " not in out[4] and "&lt;b class=&quot;x&quot;&gt;" in out[4]
        assert out[5] == "<p>&lt;span&gt;s&lt;/span&gt;</p>"


class TestQuotes:
    def test_a_quote_is_marked_with_its_own_marks_inside(self):
        assert one('She said "hello there" to me.') == (
            '<p>She said <span class="forge-assistant-quote">&quot;hello there&quot;</span> to me.</p>')

    def test_typographic_quotes_and_dialogue(self):
        out = one('“Curly” then "Hello," she said. "Fine?"')

        assert out.count('<span class="forge-assistant-quote">') == 3

    def test_no_quote_in_code_a_link_or_a_measurement(self):
        out = draw('`"code"`', '[a](https://x.y "title")', '5" tall', '```\n"fenced"\n```')

        assert all("forge-assistant-quote" not in html for html in out)

    def test_a_quote_inside_emphasis_and_emphasis_inside_a_quote(self):
        assert one('**"Hello," she said.**') == (
            '<p><strong><span class="forge-assistant-quote">&quot;Hello,&quot;</span> she said.</strong></p>')
        assert one('"a *b* c"') == (
            '<p><span class="forge-assistant-quote">&quot;a <em>b</em> c&quot;</span></p>')

    def test_a_quote_that_crosses_an_emphasis_is_left_unmarked_and_the_html_whole(self):
        out = one('**"bold** text"')

        assert "forge-assistant-quote" not in out
        assert out == "<p><strong>&quot;bold</strong> text&quot;</p>"

    def test_a_quote_and_an_inline_tag_that_cross_are_paired_each_with_its_own(self):
        """A closing tag pairs with its opening tag and a closing quote with its
        opening quote, never one with the other."""
        assert one('<b>"a</b> b"') == "<p><b>&quot;a</b> b&quot;</p>"
        assert one('"a <b>b" c</b>') == "<p>&quot;a <b>b&quot; c</b></p>"
        # A tag left open inside one that closed is its text, never dropped.
        assert one("<b><i>x</b></i>") == "<p><b>&lt;i&gt;x</b>&lt;/i&gt;</p>"

    def test_a_quotation_is_a_span_never_a_q_element(self):
        """A Gecko browser's plain-text copy writes a quote mark at each end of
        a <q> on top of the message's own, so a selection copied from LibreWolf
        turned "Hello" into ""Hello"" (2026-11-06; checked in a Gecko build):
        a quotation is a span, and no reply ever comes out with a <q>."""
        out = draw('Character stepped in the room "Hello" and waves.', '“Curly” and «guillemets»',
                   '> "quoted" in a block quote', '- "a list item"', '<q>bare</q> "x"')

        assert out[0] == ('<p>Character stepped in the room <span class="forge-assistant-quote">'
                          '&quot;Hello&quot;</span> and waves.</p>')
        assert all("<q" not in html for html in out)

    def test_a_quote_in_a_table_cell(self):
        assert '<td><span class="forge-assistant-quote">&quot;q&quot;</span></td>' in one(
            '| a |\n|---|\n| "q" |')


class TestMath:
    def test_before_katex_has_loaded_a_formula_is_shown_as_written(self):
        out = one("Then $\\rightarrow$ go")

        assert out == ('<p>Then <span class="forge-assistant-math" data-tex="\\rightarrow" '
                       'data-display="0">$\\rightarrow$</span> go</p>')

    def test_the_tex_waiting_in_its_attribute_is_escaped(self):
        out = one('$a<b \\text{"q"}$')

        assert 'data-tex="a&lt;b \\text{&quot;q&quot;}"' in out
        assert "<b" not in out.replace("<br", "")

    def test_with_katex_gemmas_arrow_is_typeset(self):
        out = one("Then $\\rightarrow$ go", katex=True)

        assert out.startswith("<p>Then <span class=\"katex\">")
        assert '<annotation encoding="application/x-tex">\\rightarrow</annotation>' in out
        assert "forge-assistant-math" not in out and "$" not in out

    def test_every_way_of_writing_math_is_typeset(self):
        out = draw("$x^2$", "$$E=mc^2$$", "\\(a\\)", "\\[b\\]",
                   "\\begin{pmatrix} 1 & 2 \\\\ 3 & 4 \\end{pmatrix}", katex=True)

        assert all('class="katex"' in html for html in out)
        assert [html.count("katex-display") for html in out] == [0, 1, 0, 1, 1]

    def test_a_formula_alone_in_its_paragraph_is_displayed_without_one(self):
        out = one("Before\n\n$$\n\\frac{a}{b}\n$$\n\nAfter", katex=True)

        assert out.startswith('<p>Before</p><span class="katex-display">')
        assert out.endswith("<p>After</p>")

    def test_an_address_right_before_a_formula_stops_at_the_formula(self):
        """A bare address runs to the next space; it must not run on into the
        formula drawn after it (a seeded fuzz found it carrying one away)."""
        out = one("see https://a.b/c$x^2$ now", katex=True)

        assert '<a href="https://a.b/c" target="_blank" rel="noopener noreferrer">https://a.b/c</a>' in out
        assert '<annotation encoding="application/x-tex">x^2</annotation>' in out

    def test_a_formula_the_reading_found_where_a_list_holds_code_is_shown_as_written(self):
        """The reading takes an indented block below a list item for the
        item's own text, as LLM Studio's does; the item's blocks then make it
        code, and code shows what was written."""
        out = one("- item\n\n        $x$ in code", katex=True)

        assert "$x$ in code</code></pre>" in out
        assert 'class="katex"' not in out

    def test_math_in_code_is_shown_as_written(self):
        out = one("`$x$` and\n```\n$$y$$\n```", katex=True)

        assert "<code>$x$</code>" in out and "<code>$$y$$</code>" in out
        assert 'class="katex"' not in out

    def test_money_stays_money(self):
        assert one("It costs $5 and $10.", katex=True) == "<p>It costs $5 and $10.</p>"

    def test_katex_with_trust_off_makes_no_link_class_or_picture_a_formula_asks_for(self):
        out = draw("$\\href{javascript:alert(1)}{x}$", "$\\url{https://a.b}$",
                   "$\\htmlClass{evil}{x}$", "$\\htmlStyle{position:fixed}{x}$",
                   "$\\includegraphics{https://a.b/c.png}$", katex=True)

        for html in out:
            assert "href=" not in html and "<img" not in html and "<a " not in html
            styles = re.findall(r'style="([^"]*)"', html)
            assert all("position" not in style for style in styles), styles
            assert 'class="evil' not in html and ' evil"' not in html

    def test_a_formula_cannot_make_a_bubble_wider_than_twenty_ems(self):
        out = one("$\\rule{5000em}{1em}\\kern{9000em}$", katex=True)

        sizes = [float(size) for size in re.findall(r"(?:width|margin-right):([\d.]+)em", out)]
        assert sizes and max(sizes) <= 20

    def test_the_same_formula_is_typeset_once(self):
        found = run(f"""
            const fs = await import("fs");
            const holder = {{exports: {{}}}};
            new Function("module", "exports", "define",
                         fs.readFileSync({json.dumps(str(KATEX))}, "utf8"))(holder, holder.exports, undefined);
            let calls = 0;
            const counting = {{renderToString(...args) {{ calls += 1;
                                                       return holder.exports.renderToString(...args); }}}};
            NS.useKatex(counting);
            NS.renderMarkdown("$x$ and $x$"); NS.renderMarkdown("$x$ again");
            console.log(JSON.stringify({{calls}}));
        """, sources=("markdown",))

        assert found["calls"] == 1


class TestTheLoader:
    LOADER = """
        const fs = await import("fs");
        const source = fs.readFileSync(%s, "utf8");
        const fetched = [];
        const linked = [];
        let answer = "ok";
        globalThis.fetch = (url) => {
            fetched.push(url);
            if (answer === "refuse") return Promise.resolve({ok: false, status: 404});
            return Promise.resolve({ok: true, text: () => Promise.resolve(source)});
        };
        document.head = {appendChild(node) { linked.push(node.href);
                                              if (answer === "refuse") node.onerror(); else node.onload(); }};
        globalThis.location.pathname = "/forge/";
        NS.basePath = () => "/forge";
    """ % json.dumps(str(KATEX))

    def test_the_first_formula_fetches_katex_and_the_next_message_is_typeset(self):
        found = run(self.LOADER + """
            const before = NS.renderMarkdown("$x$");
            const lib = await NS.loadKatex();
            const after = NS.renderMarkdown("$x$");
            console.log(JSON.stringify({before, after, fetched, linked,
                                        loaded: !!(lib && lib.renderToString)}));
        """, sources=("markdown",))

        assert "forge-assistant-math" in found["before"]
        assert found["loaded"] and 'class="katex"' in found["after"]
        assert found["fetched"] == ["/forge/model-chain/katex/0.16.10/katex.min.js"]
        assert found["linked"] == ["/forge/model-chain/katex/0.16.10/katex.min.css"]

    def test_a_message_without_math_fetches_nothing(self):
        found = run(self.LOADER + """
            NS.renderMarkdown("No math, $5 and $10, `$x$`.");
            await Promise.resolve();
            console.log(JSON.stringify({fetched, linked}));
        """, sources=("markdown",))

        assert found == {"fetched": [], "linked": []}

    def test_a_failed_fetch_leaves_formulas_as_written_and_is_not_retried_at_once(self):
        found = run(self.LOADER + """
            answer = "refuse";
            const warned = [];
            console.warn = (...args) => warned.push(String(args[0]));
            NS.renderMarkdown("$x$");
            const first = await NS.loadKatex();
            const again = NS.loadKatex();
            const shown = NS.renderMarkdown("$y$");
            console.log(JSON.stringify({first, again, fetched: fetched.length, shown, warned}));
        """, sources=("markdown",))

        assert found["first"] is None and found["again"] is None
        assert found["fetched"] == 1, "a minute before it is asked for again"
        assert "$y$</span>" in found["shown"]
        assert found["warned"] and "KaTeX did not load" in found["warned"][0]


class TestSafety:
    ALLOWED = {
        "p": set(), "br": set(), "h1": set(), "h2": set(), "h3": set(), "h4": set(), "h5": set(),
        "h6": set(), "ul": set(), "ol": {"start"}, "li": {"class"},
        "span": {"class", "aria-hidden", "data-tex", "data-display"}, "blockquote": set(),
        "hr": set(), "pre": {"class"}, "code": set(), "strong": set(), "em": set(), "del": set(),
        "a": {"href", "target", "rel"}, "table": set(), "thead": set(),
        "tbody": set(), "tr": set(), "th": {"class"}, "td": {"class"}, "b": set(), "i": set(),
        "u": set(), "s": set(), "ins": set(), "mark": set(), "sub": set(), "sup": set(),
        "small": set(), "kbd": set(),
    }

    def test_nothing_a_reply_writes_comes_out_as_an_element_or_attribute_of_its_own(self):
        """Seeded soups of Markdown, HTML, entities, addresses and this file's
        own private marks: every tag drawn is on the list, every attribute is
        one that tag may have, every address passes safeUrl, and no private
        mark is left in the HTML."""
        tokens = ["$", "$$", "\\(", "\\)", "`", "```", '"', "“", "”", "|", "\n", "\n\n",
                  "x", " ", "\\", "~~", "~", "<b>", "</b>", "<i>", "<script>", "</script>",
                  "<img src=x onerror=alert(1)>", '<a href="javascript:x">', "<br>", "<br/>",
                  "](", ")", "[", "]", "![", "javascript:alert(1)", "https://a.b/c", "www.x.org",
                  "data:text/html,<x>", "//evil.x", "<https://ok.x>", "<javascript:x>",
                  "<a@b.co>", "&lt;", "&amp;", "&#60;", "&#x3c;", "&rarr;", "&bogus;",
                  '"onload="', "- ", "* ", "1. ", "3) ", "> ", "# ", "### ", "---", "***",
                  "| a | b |\n|---|:-:|\n", "    ", "- [ ] ", "- [x] ", "**", "*", "__", "_",
                  "", "", "", "", "", "\u0000", "\\begin{align}",
                  "\\end{align}", "'", "=", "<q>", "</q>", "<span class=x>", "</span>", "\t"]
        found = run("""
            const tokens = %s;
            let seed = 20261105;
            const chance = () => { seed = (seed * 1103515245 + 12345) & 0x7fffffff; return seed / 0x7fffffff; };
            const out = [];
            for (let n = 0; n < 1500; n += 1) {
                let text = "";
                const length = 1 + Math.floor(chance() * 50);
                for (let k = 0; k < length; k += 1) text += tokens[Math.floor(chance() * tokens.length)];
                out.push(NS.renderMarkdown(text));
            }
            console.log(JSON.stringify({out}));
        """ % json.dumps(tokens), sources=("markdown",))

        problems = []
        drawn = 0
        for html in found["out"]:
            if re.search("[-\u0000]", html):
                problems.append(("mark", html))
            for tag in re.findall(r"<[^<>]*>", html):
                shape = re.fullmatch(r'<(/?)([a-z0-9]+)((?:\s+[a-z-]+="[^"<>]*")*)\s*>', tag)
                if not shape or shape.group(2) not in self.ALLOWED:
                    problems.append(("tag", tag))
                    continue
                for name, value in re.findall(r'([a-z-]+)="([^"]*)"', shape.group(3)):
                    if name not in self.ALLOWED[shape.group(2)]:
                        problems.append(("attribute", tag))
                    if name == "href" and not re.match(r"(?i)(https?:|mailto:|/(?!/))", value):
                        problems.append(("href", tag))
                drawn += 1
        assert drawn > 10000
        assert not problems, problems[:5]

    def test_this_files_own_marks_in_a_reply_are_replaced_before_anything_is_read(self):
        out = one("0 q \u0000")

        assert out == "<p>�0� �q� �</p>"

    def test_safe_url_refuses_what_cannot_be_seen(self):
        found = run("""
            console.log(JSON.stringify({tab: NS.safeUrl("https://a.b/\\tx"),
                                        mark: NS.safeUrl("https://a.b/\\ue000"),
                                        plain: NS.safeUrl("https://a.b/x")}));
        """, sources=("markdown",))

        assert found == {"tab": "", "mark": "", "plain": "https://a.b/x"}

    def test_a_long_reply_is_drawn_in_milliseconds(self):
        """Every frame of a streaming reply draws it again."""
        found = run("""
            let long = "";
            for (let i = 0; i < 200; i += 1) {
                long += "*   **Point " + i + ":** text with $\\\\rightarrow$, \\"a quote\\" and `code`.\\n";
            }
            NS.renderMarkdown(long);
            const start = Date.now();
            for (let i = 0; i < 10; i += 1) NS.renderMarkdown(long);
            console.log(JSON.stringify({ms: (Date.now() - start) / 10, size: long.length}));
        """, sources=("markdown",))

        assert found["size"] > 12000
        assert found["ms"] < 100
