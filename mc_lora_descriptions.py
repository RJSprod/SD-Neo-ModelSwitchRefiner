"""Civitai model descriptions: HTML in, readable text out.

Civitai serves a model's "about this model" as HTML written in its own editor.
Two things make storing that markup verbatim a bad idea: the library format is
CSV, and the details panel renders into the WebUI's own dialog. So the markup is
converted **once, at fetch time**, into text that keeps the structure worth
keeping - paragraphs, headings, list bullets and link targets - and carries no
tags that anything downstream could re-interpret.

The conversion is lossy in one direction only: it never invents text. Images are
dropped (the panel cannot show a remote image inline anyway), scripts and styles
are dropped, and everything else becomes plain text.
"""

from __future__ import annotations

import html as html_module
import re
from html.parser import HTMLParser
from urllib.parse import urljoin

CIVITAI_BASE = "https://civitai.com/"

# Longest description kept. Civitai documents no ceiling and a handful of model
# pages are enormous; a cap keeps one page from dominating the CSV.
MAX_DESCRIPTION_CHARS = 20000

# Content of these never belongs in the text.
SKIP_TAGS = {"script", "style", "head", "title", "noscript", "iframe", "svg", "template"}

# Tags that end a line.
LINE_TAGS = {"li", "tr", "dt", "dd", "figcaption"}

# Tags that end a block: a blank line follows them.
BLOCK_TAGS = {
    "p",
    "div",
    "section",
    "article",
    "header",
    "footer",
    "blockquote",
    "pre",
    "table",
    "thead",
    "tbody",
    "ul",
    "ol",
    "dl",
    "hr",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
}

NO_BREAK = 0
LINE_BREAK = 1
BLOCK_BREAK = 2

_MARKUP_RE = re.compile(r"<[a-zA-Z!/]")
_WHITESPACE_RE = re.compile(r"\s+")
_INLINE_SPACE_RE = re.compile(r"[ \t]+")
_BLANK_LINES_RE = re.compile(r"\n{3,}")
_TAG_RE = re.compile(r"<[^>]*>")
# For the fallback only: the parser skips these elements' content, and stripping
# their tags alone would leave a script's source behind as "text".
_SKIPPED_BLOCK_RE = re.compile(
    r"<\s*(" + "|".join(sorted(SKIP_TAGS)) + r")\b[^>]*>.*?<\s*/\s*\1\s*>",
    re.IGNORECASE | re.DOTALL,
)


def _absolute(href: str) -> str:
    """Civitai writes site-relative links; make them usable outside the site."""
    href = href.strip()
    if not href:
        return ""
    if href.startswith(("http://", "https://")):
        return href
    if href.startswith("/"):
        return urljoin(CIVITAI_BASE, href)
    return ""


class _TextExtractor(HTMLParser):
    """Turns a fragment of Civitai HTML into text, structure intact.

    Line breaks are *requested* rather than written: the strongest break asked
    for since the last piece of text is the one that lands. That is what keeps
    ``</li><li>`` - two break requests in a row - from putting a blank line
    between every bullet.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        # HTMLParser keeps its own state on ``self`` - Python 3.13.x added a
        # ``_pending`` buffer - so everything here carries a ``_text_`` prefix.
        # Shadowing one of its names breaks ``close()`` for every description.
        self._text_parts: list[str] = []
        self._text_break = NO_BREAK
        self._text_skip = 0
        # (index into _text_parts where the link's text starts, resolved href)
        self._text_links: list[tuple[int, str]] = []
        # One counter per open list; None marks an unordered list.
        self._text_lists: list[int | None] = []

    # ------------------------------------------------------------- assembly

    def _break(self, strength: int) -> None:
        if self._text_parts:  # never open the text with blank lines
            self._text_break = max(self._text_break, strength)

    def _write(self, text: str, *, strip_leading: bool = True) -> None:
        """Append text, flushing whatever break was requested before it."""
        if self._text_break:
            if strip_leading:
                text = text.lstrip(" ")
                if not text:
                    return
            self._text_parts.append("\n" if self._text_break == LINE_BREAK else "\n\n")
            self._text_break = NO_BREAK
        elif strip_leading and not self._text_parts:
            text = text.lstrip(" ")
        if text:
            self._text_parts.append(text)

    def _bullet(self) -> None:
        self._break(LINE_BREAK)
        indent = "  " * max(0, len(self._text_lists) - 1)
        if self._text_lists and self._text_lists[-1] is not None:
            ordinal = self._text_lists[-1] + 1
            self._text_lists[-1] = ordinal
            self._write(f"{indent}{ordinal}. ", strip_leading=False)
        else:
            self._write(f"{indent}• ", strip_leading=False)

    # -------------------------------------------------------------- parsing

    def handle_starttag(self, tag: str, attrs) -> None:
        tag = tag.lower()

        if tag in SKIP_TAGS:
            self._text_skip += 1
            return
        if self._text_skip:
            return

        if tag == "br":
            self._break(LINE_BREAK)
        elif tag == "img":
            pass  # a remote image is nothing this panel can show
        elif tag in ("ul", "ol"):
            # A list nested inside an item continues that item, so it must not
            # be separated from it by a blank line.
            self._break(LINE_BREAK if self._text_lists else BLOCK_BREAK)
            self._text_lists.append(0 if tag == "ol" else None)
        elif tag == "li":
            self._bullet()
        elif tag in ("td", "th"):
            # Mid-row: no break is pending, so this is not the first cell.
            if not self._text_break and self._text_parts:
                self._write(" | ")
        elif tag == "a":
            href = ""
            for key, value in attrs:
                if key and key.lower() == "href" and value:
                    href = _absolute(value)
                    break
            self._text_links.append((len(self._text_parts), href))
        elif tag in LINE_TAGS:
            self._break(LINE_BREAK)
        elif tag in BLOCK_TAGS:
            self._break(BLOCK_BREAK)

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()

        if tag in SKIP_TAGS:
            self._text_skip = max(0, self._text_skip - 1)
            return
        if self._text_skip:
            return

        if tag == "a":
            self._close_link()
        elif tag in ("ul", "ol"):
            if self._text_lists:
                self._text_lists.pop()
            self._break(LINE_BREAK if self._text_lists else BLOCK_BREAK)
        elif tag in LINE_TAGS:
            self._break(LINE_BREAK)
        elif tag in BLOCK_TAGS:
            self._break(BLOCK_BREAK)

    def _close_link(self) -> None:
        if not self._text_links:
            return
        start, href = self._text_links.pop()
        if not href:
            return
        text = "".join(self._text_parts[start:]).strip()
        # Only spell the target out when the link text does not already carry it.
        if not text:
            self._write(href)
        elif href not in text:
            self._write(f" ({href})")

    def handle_data(self, data: str) -> None:
        if self._text_skip or not data:
            return
        # Source newlines are formatting, not content: only tags break lines.
        self._write(_WHITESPACE_RE.sub(" ", data))

    def text(self) -> str:
        return "".join(self._text_parts)


def _normalize_line(line: str) -> str:
    """Collapse runs of spaces inside a line, but keep its indentation."""
    body = line.lstrip(" \t")
    indent = line[: len(line) - len(body)].replace("\t", "  ")
    return (indent + _INLINE_SPACE_RE.sub(" ", body)).rstrip()


def _normalize(text: str) -> str:
    """Tidy whitespace without losing paragraphs or list indentation."""
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\xa0", " ")
    lines = [_normalize_line(line) for line in text.split("\n")]
    return _BLANK_LINES_RE.sub("\n\n", "\n".join(lines)).strip()


def html_to_text(value, *, limit: int = MAX_DESCRIPTION_CHARS) -> str:
    """Convert a Civitai description to text. Non-HTML input is just tidied.

    ``limit`` caps the result; a truncated description ends in an ellipsis so it
    is obvious in the panel that the model page has more.
    """
    if value is None:
        return ""

    raw = str(value)
    if not raw.strip():
        return ""

    if _MARKUP_RE.search(raw):
        parser = _TextExtractor()
        try:
            parser.feed(raw)
            parser.close()
            text = _normalize(parser.text())
        except Exception:
            # A fragment we cannot parse is worth less than the rest of the row:
            # fall back to stripped tags rather than failing the whole lookup.
            text = _normalize(html_module.unescape(_TAG_RE.sub(" ", _SKIPPED_BLOCK_RE.sub(" ", raw))))
    else:
        text = _normalize(html_module.unescape(raw))

    if limit > 0 and len(text) > limit:
        text = text[: max(0, limit - 1)].rstrip() + "…"
    return text


def summarize(text, limit: int = 240) -> str:
    """A one-line gist of a description, for summaries and tooltips."""
    flat = " ".join(str(text or "").split())
    if len(flat) <= limit:
        return flat
    return flat[: max(0, limit - 1)].rstrip() + "…"
