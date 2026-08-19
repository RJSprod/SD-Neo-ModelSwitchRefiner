"""A YAML subset, exactly wide enough for a character file.

Character files are oobabooga's own — a flat mapping of a few string fields,
written as ``key: value`` with block scalars for the multi-line ones. That is a
small enough language to read and write here rather than to add PyYAML to a
requirements file the one-click installer pins, verifies and installs by hash:
five packages that every user downloads is not a list to grow for a flat map of
strings.

Supported, which is everything the format uses:

- ``key: value``, plain, ``'single quoted'`` or ``"double quoted"``
- block scalars — ``|``, ``|-``, ``|+``, ``>``, ``>-``, ``>+`` — including the
  folded forms, where single newlines become spaces and blank lines newlines
- ``#`` comment lines, ``---``/``...`` document markers, and CRLF line endings

Not supported: sequences, anchors, aliases, multiple documents and nested
mappings. A nested mapping is skipped rather than guessed at, so a file
carrying one still loads with every scalar field it has intact.

Everything read is a string, and everything written is written so that it reads
back as the same string: a value that would otherwise parse as a number, a
boolean or ``null`` is quoted. What this writes, PyYAML reads — which is the
half of the compatibility that matters, since oobabooga is what opens these
files next.
"""

from __future__ import annotations

# Values YAML would hand back as something other than the string that was
# written. Quoting them on the way out is what keeps a character named "No" a
# string rather than False.
_KEYWORDS = {"true", "false", "yes", "no", "on", "off", "null", "none", "~"}

# A plain scalar may not open with one of these: YAML reads them as structure.
_INDICATORS = set("-?:,[]{}#&*!|>'\"%@`")


def loads(text: str) -> dict[str, str]:
    """Every top-level ``key: value`` in ``text``, as strings."""
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    values: dict[str, str] = {}
    index = 0
    while index < len(lines):
        line, index = lines[index], index + 1
        stripped = line.strip()
        # Blank lines, comments, document markers, and any line still indented
        # under a key whose block we chose not to read.
        if not stripped or stripped.startswith("#") or stripped in ("---", "..."):
            continue
        if line[:1].isspace():
            continue
        key, separator, rest = line.partition(":")
        key, rest = key.strip(), rest.strip()
        if not separator or not key:
            continue
        if rest[:1] in ("|", ">"):
            block, index = _indented_block(lines, index)
            values[key] = _fold(block) if rest[0] == ">" else block
        elif rest == "":
            # ``key:`` with an indented block under it is either a nested
            # mapping — which this does not pretend to parse — or a block
            # scalar that forgot its marker.
            block, index = _indented_block(lines, index)
            values[key] = "" if _looks_like_mapping(block) else block
        else:
            values[key] = _scalar(rest)
    return values


def dumps(values: dict[str, object]) -> str:
    """``values`` as YAML, in the order given."""
    out = []
    for key, value in values.items():
        text = "" if value is None else str(value)
        if "\n" in text:
            body = "\n".join(("  " + line).rstrip() for line in text.split("\n"))
            out.append(f"{key}: |-\n{body}\n")
        else:
            out.append(f"{key}: {_quote(text)}\n")
    return "".join(out)


def _indented_block(lines: list[str], index: int) -> tuple[str, int]:
    """The run of indented (or blank) lines at ``index``, dedented."""
    collected: list[str] = []
    while index < len(lines):
        line = lines[index]
        if line.strip() and not line[:1].isspace():
            break
        collected.append(line)
        index += 1
    while collected and not collected[-1].strip():
        collected.pop()
    if not collected:
        return "", index
    indent = min(len(line) - len(line.lstrip()) for line in collected if line.strip())
    return "\n".join(line[indent:].rstrip() for line in collected), index


def _fold(block: str) -> str:
    """A folded scalar: single newlines are spaces, blank lines are newlines."""
    out, pending = "", ""
    for line in block.split("\n"):
        if not line.strip():
            out, pending = out + "\n", ""
            continue
        out += (pending + line.strip()) if out and not out.endswith("\n") else line.strip()
        pending = " "
    return out.strip("\n")


def _looks_like_mapping(block: str) -> bool:
    lines = [line for line in block.split("\n") if line.strip()]
    return bool(lines) and all(
        ":" in line and not line.lstrip().startswith("-") for line in lines[:2])


def _scalar(text: str) -> str:
    if len(text) >= 2 and text[0] == text[-1] and text[0] in ("'", '"'):
        body = text[1:-1]
        if text[0] == "'":
            return body.replace("''", "'")
        return (body.replace("\\n", "\n").replace("\\t", "\t")
                    .replace('\\"', '"').replace("\\\\", "\\"))
    return text


def _quote(text: str) -> str:
    plain = (text and text == text.strip() and text[0] not in _INDICATORS
             and ": " not in text and " #" not in text
             and text.casefold() not in _KEYWORDS and not _numeric(text))
    if plain:
        return text
    escaped = (text.replace("\\", "\\\\").replace('"', '\\"')
                   .replace("\n", "\\n").replace("\t", "\\t"))
    return f'"{escaped}"'


def _numeric(text: str) -> bool:
    try:
        float(text)
    except ValueError:
        return False
    return True
