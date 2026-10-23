"""HTML fragments for the details panel and the extension tab.

Everything here styles itself with Gradio's own CSS custom properties
(``--body-text-color``, ``--border-color-primary``, ...) so a user's theme keeps
applying; there are no hard-coded colours.
"""

from __future__ import annotations

import html
import re
from urllib.parse import urlparse

import mc_lora_layouts
from mc_lora_common import truncate

MAX_VALUE_CHARS = 4000

# Applied to already-escaped text, so `"` and `<` cannot appear inside a match;
# a trailing `.`/`)` is far more likely to be sentence punctuation than URL.
URL_IN_TEXT_RE = re.compile(r"https?://[^\s<>()\[\]&]+(?:&amp;[^\s<>()\[\]&]*)*")
PARAGRAPH_SPLIT_RE = re.compile(r"\n{2,}")


def escape(value) -> str:
    return html.escape(str(value if value is not None else ""), quote=True)


def _linkify(value: str) -> str:
    text = str(value).strip()
    if text.startswith(("http://", "https://")) and " " not in text:
        return f'<a href="{escape(text)}" target="_blank" rel="noopener noreferrer">{escape(truncate(text, 90))}</a>'
    return escape(value).replace("\n", "<br>")


def _linkify_in_text(escaped: str) -> str:
    """Turn URLs inside a run of escaped text into links."""

    def replace(match) -> str:
        url = match.group(0)
        trailing = ""
        while url and url[-1] in ".,;:!?'\"":
            trailing = url[-1] + trailing
            url = url[:-1]
        if not url:
            return match.group(0)
        return f'<a href="{url}" target="_blank" rel="noopener noreferrer">{url}</a>{trailing}'

    return URL_IN_TEXT_RE.sub(replace, escaped)


def rich_text(value: str) -> str:
    """Escaped text with its paragraphs, line breaks and links preserved.

    The library stores descriptions as text, never as markup, so this is the one
    place structure is turned back into HTML - from characters the converter put
    there, not from anything Civitai wrote.
    """
    text = str(value or "").strip()
    if not text:
        return ""
    paragraphs = []
    for block in PARAGRAPH_SPLIT_RE.split(text):
        lines = [_linkify_in_text(escape(line)) for line in block.split("\n")]
        paragraphs.append("<p>" + "<br>".join(lines) + "</p>")
    return "".join(paragraphs)


def notice(message: str, *, kind: str = "info") -> str:
    return f'<div class="mc-lora-notice mc-lora-notice-{escape(kind)}">{message}</div>'


# ------------------------------------------------------------ panel pieces
#
# The panel reads like the Inspect view of the Wan2GP LoRA browser: a title
# strip, a sidebar with the facts, trigger words and texts, and the gallery
# prompts beside it. Everything clickable carries a ``data-mc-lora-action`` that
# javascript/extended-lora-details.js handles; no inline script is emitted.

# A description longer than this is shown clamped, with a "Show more" toggle.
# The page un-clamps one that turns out to fit (see the script's mcLoraFitClamps).
CLAMP_CHARS = 900
CLAMP_LINES = 14


def tool_button(label: str, action: str, *, title: str = "") -> str:
    tip = f' title="{escape(title)}"' if title else ""
    return f'<button type="button" class="mc-lora-tool" data-mc-lora-action="{escape(action)}"{tip}>{escape(label)}</button>'


def _source(text: str) -> str:
    """The exact text a section's Copy/Use buttons act on, kept out of sight."""
    return f'<textarea class="mc-lora-copy-source" hidden aria-hidden="true" tabindex="-1">{escape(text)}</textarea>'


def section_html(title: str, body: str, *, count=None, note: str = "", tools: str = "", classes: str = "", attrs: str = "", source=None) -> str:
    counter = f' <span class="mc-lora-count">{escape(count)}</span>' if count is not None else ""
    aside = f' <span class="mc-lora-muted">{note}</span>' if note else ""
    head = (
        '<div class="mc-lora-section-head">'
        f'<span class="mc-lora-section-title">{escape(title)}{counter}{aside}</span>'
        f'<span class="mc-lora-section-tools">{tools}</span>'
        "</div>"
    )
    hidden = _source(source) if source is not None else ""
    return f'<section class="mc-lora-section {classes}"{attrs}>{head}{body}{hidden}</section>'


def _pill(text: str, *, accent: bool = False) -> str:
    return f'<span class="mc-lora-pill{" mc-lora-pill-accent" if accent else ""}">{escape(text)}</span>'


def _record_link(record) -> str:
    title = escape(record.title)
    if record.url:
        return f'<a href="{escape(record.url)}" target="_blank" rel="noopener noreferrer">{title}</a>'
    return title


def header_html(*, records, how: str) -> str:
    """The title strip: the model's name, its version, base model and creator."""
    if not records:
        return notice(
            "No CSV row matched this LoRA. Fetch it from Civitai with the button below, or add a CSV to the "
            "library in <b>Settings → Model Chain: LoRA Details</b>.",
            kind="empty",
        )

    places = ", ".join(f"{escape(record.source)} (row {record.row_number})" for record in records)
    resolved = [record for record in records if record.resolved]
    if not resolved:
        # A fetch records the LoRAs Civitai had nothing for too: say so, and why.
        status = ", ".join(sorted({record.status for record in records if record.status})) or "no details"
        reason = next((record.note for record in records if record.note), "")
        detail = f' — <span title="{escape(reason)}">{escape(truncate(reason, 280))}</span>' if reason else ""
        return notice(
            f"Civitai had nothing for this file when it was last fetched (<code>{escape(status)}</code>){detail}.",
            kind="empty",
        ) + f'<div class="mc-lora-head-sub">Recorded in {places}</div>'

    primary = resolved[0]
    names: list[str] = []
    for record in resolved:
        link = _record_link(record)
        if link not in names:
            names.append(link)

    pills = []
    if primary.version_name:
        pills.append(_pill(primary.version_name))
    if primary.base_model:
        pills.append(_pill(primary.base_model, accent=True))
    if primary.creator:
        pills.append(_pill(f"by {primary.creator}"))

    action = ""
    if primary.url:
        action = (
            f'<a class="mc-lora-open-link" href="{escape(primary.url)}" target="_blank" rel="noopener noreferrer">'
            "Open on Civitai ↗</a>"
        )

    sub = f'Matched by {escape(how)} in {places}'

    strip = (
        '<div class="mc-lora-head">'
        '<div class="mc-lora-head-text">'
        f'<div class="mc-lora-head-title">{" · ".join(names)}</div>'
        f'<div class="mc-lora-pills">{"".join(pills)}</div>'
        "</div>"
        f"{action}"
        "</div>"
        f'<div class="mc-lora-head-sub">{sub}</div>'
    )
    return strip


def facts_html(records, *, identity: dict, architecture: str = "") -> str:
    """Key facts, Wan2GP-style: small keys, values beside them."""
    rows: list[tuple[str, str]] = []
    primary = records[0] if records else None
    if primary is not None:
        for label, value in (
            ("Version", primary.version_name),
            ("Base model", primary.base_model),
            ("Creator", primary.creator),
        ):
            if value:
                rows.append((label, escape(value)))
        ids = " · ".join(
            f"{name} {escape(value)}" for name, value in (("model", primary.model_id), ("version", primary.version_id)) if value
        )
        if ids:
            rows.append(("Civitai", ids))
        if primary.status and primary.status != "ok":
            tip = f' title="{escape(primary.note)}"' if primary.note else ""
            rows.append(("Status", f"<span{tip}>{escape(primary.status)}</span>"))
        if len(records) > 1:
            rows.append(("Rows", f"{len(records)} matched — every field is under <i>All fields</i>"))

    if architecture:
        rows.append(("Detected", f"{escape(architecture)} <span class=\"mc-lora-muted\">from its tensor names</span>"))
    for label, value in identity.items():
        if value:
            rows.append((label, f'<code class="mc-lora-hash">{escape(value)}</code>'))

    if not rows:
        return ""
    body = "".join(f'<div class="mc-lora-fact"><span class="mc-lora-fact-key">{escape(label)}</span><span class="mc-lora-fact-value">{value}</span></div>' for label, value in rows)
    return f'<div class="mc-lora-facts">{body}</div>'


def triggers_html(words, *, target: str = "", active=()) -> str:
    """Trigger words as chips. With a target, a chip toggles the word in the
    dialog's activation text; without one, it copies the word."""
    words = list(words or ())
    if not words:
        return ""
    on = set(active or ())
    action = "toggle-trigger" if target else "copy-word"
    tip = "Add to or remove from the activation text" if target else "Copy"
    chips = "".join(
        f'<button type="button" class="mc-lora-word{" mc-lora-word-on" if word in on else ""}" data-mc-lora-action="{action}" '
        f'data-word="{escape(word)}" title="{tip}">{escape(word)}</button>'
        for word in words
    )
    tools = tool_button("Copy all", "copy")
    if target:
        tools += tool_button("Use", "use-triggers", title="Use these as the activation text")
        tools += tool_button("Append", "append-triggers", title="Add the missing ones to the activation text")
    attrs = f' data-mc-lora-target="{escape(target)}"' if target else ""
    return section_html(
        "Trigger words",
        f'<div class="mc-lora-words">{chips}</div>',
        count=len(words),
        tools=tools,
        classes="mc-lora-triggers",
        attrs=attrs,
        source=", ".join(words),
    )


def text_section_html(title: str, text: str, *, note: str = "") -> str:
    """A copyable block of text - a description, version notes - clamped when long."""
    text = str(text or "").strip()
    if not text:
        return ""
    long = len(text) > CLAMP_CHARS or text.count("\n") > CLAMP_LINES
    body = f'<div class="mc-lora-text mc-lora-description{" mc-lora-clamp" if long else ""}">{rich_text(text)}</div>'
    if long:
        body += '<button type="button" class="mc-lora-more-toggle" data-mc-lora-action="expand" aria-expanded="false">Show more</button>'
    return section_html(title, body, note=note, tools=tool_button("Copy", "copy"), classes="mc-lora-text-section", source=text)


def descriptions_html(blocks) -> str:
    """The model description and version notes of every matched row."""
    if not blocks:
        return ""
    several = len(blocks) > 1
    parts = []
    for title, source, model_text, version_text in blocks:
        note = escape(f"{title} · {source}") if several else ""
        parts.append(text_section_html("Description", model_text, note=note))
        parts.append(text_section_html("Version notes", version_text, note=note))
    return "".join(parts)


def negative_html(negatives, *, tab: str) -> str:
    text = "\n".join(value for value in (negatives or ()) if value)
    if not text:
        return ""
    tools = tool_button("Copy", "copy") + tool_button("Append to negative", "append-negative", title="Append to the negative prompt")
    body = f'<div class="mc-lora-text mc-lora-prompt-text">{escape(text)}</div>'
    return section_html("Negative prompt", body, tools=tools, attrs=f' data-mc-lora-tab="{escape(tab)}"', source=text)


def _value_cell(value: str) -> str:
    text = str(value)
    if len(text) > MAX_VALUE_CHARS:
        text = text[:MAX_VALUE_CHARS] + "…"
    return _linkify(text)


def fields_html(records) -> str:
    """Every field of the matched row(s), extra columns included, folded away."""
    if not records:
        return ""

    blocks = []
    for record in records:
        fields: list[tuple[str, str]] = []
        if record.model_name:
            fields.append(("Model", record.model_name))
        if record.url:
            fields.append(("Civitai", record.url))
        if record.trigger_words:
            fields.append(("Trigger words", ", ".join(record.trigger_words)))
        fields.append(("Prompts", str(len(record.prompts))))
        if record.negative_prompts:
            fields.append(("Negative prompt", "\n".join(record.negative_prompts)))
        if record.rel_path:
            fields.append(("CSV file path", record.rel_path))
        if record.sha256:
            fields.append(("SHA256", record.sha256))
        for label, value in record.extras.items():
            fields.append((label, value))

        body = "".join(f"<tr><th>{escape(label)}</th><td>{_value_cell(value)}</td></tr>" for label, value in fields if str(value).strip())
        blocks.append(
            f'<div class="mc-lora-source-block">'
            f'<div class="mc-lora-source-title">{escape(record.source)} <span class="mc-lora-muted">· row {record.row_number}</span></div>'
            f'<table class="mc-lora-table">{body}</table>'
            f"</div>"
        )

    return f'<details class="mc-lora-more"><summary>All fields</summary>{"".join(blocks)}</details>'


def raw_html(raw_json: str) -> str:
    """The matched row(s) as JSON, folded away, with a Copy button."""
    if not raw_json or raw_json == "{}":
        return ""
    return (
        '<details class="mc-lora-more mc-lora-raw-wrap"><summary>Raw CSV row</summary>'
        '<section class="mc-lora-section mc-lora-raw-section">'
        f'<div class="mc-lora-section-head"><span class="mc-lora-section-title"></span><span class="mc-lora-section-tools">{tool_button("Copy", "copy")}</span></div>'
        f'<pre class="mc-lora-raw mc-lora-copy-source">{escape(raw_json)}</pre>'
        "</section></details>"
    )


def column_title_html(title: str, note: str = "") -> str:
    """A column's heading in the dialog: its name, and a word on what is under it."""
    aside = f' <span class="mc-lora-muted">{escape(note)}</span>' if note else ""
    return f'<div class="mc-lora-col-title">{escape(title)}{aside}</div>'


def prompts_head_html(total: int, saved: int) -> str:
    if total <= 0:
        return ""
    note = f"{saved} of {total} images saved" if saved else "no images saved yet"
    return (
        '<div class="mc-lora-section-head">'
        f'<span class="mc-lora-section-title">Gallery prompts <span class="mc-lora-count">{total}</span></span>'
        f'<span class="mc-lora-muted">{escape(note)}</span>'
        "</div>"
    )


def layout_html(layout) -> str:
    """What a MOSS-Image layout prompt depicts, read from its JSON."""
    if layout is None:
        return ""
    items = "".join(
        f'<li><span class="mc-lora-layout-label">{escape(item.label)}</span>{escape(item.caption)}'
        + (f' <span class="mc-lora-muted">“{escape(item.text)}”</span>' if item.text else "")
        + "</li>"
        for item in layout.items[:24]
    )
    more = f'<li class="mc-lora-muted">…and {len(layout.items) - 24} more</li>' if len(layout.items) > 24 else ""
    described = layout.describe()
    return (
        '<div class="mc-lora-layout">'
        f'<div class="mc-lora-layout-head"><span class="mc-lora-badge-inline">MOSS layout</span>{escape(described)}</div>'
        + (f'<div class="mc-lora-layout-caption">{escape(layout.caption)}</div>' if layout.caption else "")
        + (f'<ul class="mc-lora-layout-items">{items}{more}</ul>' if items else "")
        + "</div>"
    )


def prompt_meta_html(total: int, index: int, source: str = "", layout=None) -> str:
    if total <= 0:
        return notice("This LoRA has no gallery prompts in the library.", kind="empty")
    origin = f' <span class="mc-lora-muted">· {escape(source)}</span>' if source else ""
    return f'<div class="mc-lora-count-line">Prompt <b>{index + 1}</b> of <b>{total}</b>{origin}</div>{layout_html(layout)}'


def image_status_html(*, downloaded: bool, name: str, url: str) -> str:
    """The line beside the selected prompt's image."""
    source = urlparse(url).netloc if url else ""
    origin = f' <span class="mc-lora-muted">({escape(source)})</span>' if source else ""

    if downloaded:
        return notice(f"Saved as <code>{escape(name)}</code>{origin}.", kind="ok")
    return notice(f"The gallery image behind this prompt is not downloaded yet{origin}.", kind="info")


def prompt_cards_html(cards, *, total: int, tab: str, bridge: str) -> str:
    """Every gallery prompt as a card - the image when it is saved, the prompt,
    and its actions - like the media grid of Wan2GP's Inspect view.

    ``cards`` holds one dict per shown prompt: ``index``, ``prompt``, ``thumb``
    (a data URI or ``""``), ``saved``, ``video`` and ``url``.
    """
    if not cards:
        return ""

    with_media = any(card["saved"] for card in cards)
    articles = []
    for card in cards:
        number = card["index"] + 1
        layout = mc_lora_layouts.parse(card["prompt"])
        shown = layout.caption if layout is not None and layout.caption else card["prompt"]

        media = ""
        if with_media:
            if card["thumb"]:
                media = (
                    f'<button type="button" class="mc-lora-card-media" data-mc-lora-action="view" title="Show prompt {number} above">'
                    f'<img src="{card["thumb"]}" alt="" loading="lazy"></button>'
                )
            elif card["saved"]:
                label = "Video saved" if card["video"] else "Saved"
                media = f'<div class="mc-lora-card-media mc-lora-card-media-empty">{label}</div>'
            elif card["url"]:
                media = (
                    '<div class="mc-lora-card-media mc-lora-card-media-empty">'
                    '<button type="button" class="mc-lora-tool" data-mc-lora-action="download">Download image</button></div>'
                )
            else:
                media = '<div class="mc-lora-card-media mc-lora-card-media-empty">No image URL</div>'

        badge = '<span class="mc-lora-badge-inline">layout</span>' if layout is not None else ""
        tools = (
            tool_button("Copy", "copy")
            + tool_button("Append", "append-prompt", title="Append to the prompt")
            + tool_button("Replace", "replace-prompt", title="Replace the prompt")
        )
        articles.append(
            f'<article class="mc-lora-card" data-index="{card["index"]}">'
            f"{media}"
            '<div class="mc-lora-card-head">'
            f'<button type="button" class="mc-lora-card-num" data-mc-lora-action="view" title="Show this prompt above">#{number}</button>'
            f'{badge}<span class="mc-lora-section-tools">{tools}</span>'
            "</div>"
            f'<div class="mc-lora-card-text">{escape(shown)}</div>'
            f"{_source(card['prompt'])}"
            "</article>"
        )

    more = ""
    if total > len(cards):
        more = f'<div class="mc-lora-muted mc-lora-more-note">…and {total - len(cards)} more — the selector above reaches every prompt.</div>'
    return (
        '<details class="mc-lora-more mc-lora-cards-wrap" open>'
        f'<summary>All prompts <span class="mc-lora-count">{total}</span></summary>'
        f'<div class="mc-lora-cards" data-mc-lora-tab="{escape(tab)}" data-mc-lora-bridge="{escape(bridge)}">{"".join(articles)}</div>'
        f"{more}</details>"
    )


def library_summary_html(stats: dict) -> str:
    errors = stats.get("errors") or []
    rows = [
        ("Library directory", stats.get("directory", "")),
        ("CSV files", str(stats.get("files", 0))),
        ("Rows", str(stats.get("rows", 0))),
        ("Rows with a SHA256", str(stats.get("hashed_rows", 0))),
        ("Prompts", str(stats.get("prompts", 0))),
        ("Rows with a description", str(stats.get("described_rows", 0))),
    ]
    body = "".join(f"<tr><th>{escape(label)}</th><td>{escape(value)}</td></tr>" for label, value in rows)
    table = f'<table class="mc-lora-table">{body}</table>'

    if not stats.get("files"):
        table += notice("The library is empty. Upload a CSV, or run a Civitai fetch on a folder.", kind="empty")
    for source in errors:
        table += notice(f"<b>{escape(source.name)}</b>: {escape(source.error)}", kind="error")
    return table


def image_library_html(stats: dict) -> str:
    from mc_lora_common import human_bytes

    rows = [
        ("Image folder", stats.get("directory", "")),
        ("Downloaded images", str(stats.get("files", 0))),
        ("Total size", human_bytes(stats.get("bytes", 0))),
    ]
    body = "".join(f"<tr><th>{escape(label)}</th><td>{escape(value)}</td></tr>" for label, value in rows)
    table = f'<table class="mc-lora-table">{body}</table>'
    if not stats.get("files"):
        table += notice(
            "Nothing downloaded yet. Open a LoRA's details dialog and use <b>Download this image</b> "
            "beside a gallery prompt, or <b>Download image</b> on its card.",
            kind="empty",
        )
    return table


def sources_html(sources) -> str:
    if not sources:
        return ""
    from mc_lora_common import human_bytes

    rows = "".join(
        "<tr>"
        f"<td>{escape(source.name)}</td>"
        f'<td class="mc-lora-num">{source.rows}</td>'
        f'<td class="mc-lora-num">{escape(human_bytes(source.size))}</td>'
        f"<td>{escape(source.error) if source.error else ''}</td>"
        "</tr>"
        for source in sources
    )
    return (
        '<table class="mc-lora-table mc-lora-grid">'
        "<thead><tr><th>File</th><th>Rows</th><th>Size</th><th>Problem</th></tr></thead>"
        f"<tbody>{rows}</tbody></table>"
    )


def _progress_bar(percent: int) -> str:
    return f'<div class="mc-lora-bar"><div class="mc-lora-bar-fill" style="width:{max(0, min(100, percent))}%"></div></div>'


def job_html(snapshots, *, log_lines: int = 18) -> str:
    """The job monitor: newest first, with the running job's log tailed."""
    if not snapshots:
        return notice("No scans have been started yet.", kind="empty")

    state_labels = {
        "queued": "Queued",
        "running": "Running",
        "done": "Finished",
        "failed": "Failed",
        "cancelled": "Cancelled",
    }

    blocks = []
    for snapshot in snapshots:
        header = (
            f'<div class="mc-lora-job-head">'
            f'<span class="mc-lora-badge mc-lora-badge-{escape(snapshot.state)}">{escape(state_labels.get(snapshot.state, snapshot.state))}</span>'
            f'<b>#{snapshot.id} {escape(snapshot.title)}</b>'
            f'<span class="mc-lora-muted">{snapshot.elapsed:.0f}s</span>'
            f"</div>"
        )
        body = [header]
        if snapshot.active:
            body.append(_progress_bar(snapshot.percent))
        body.append(f'<div class="mc-lora-muted mc-lora-job-msg">{escape(snapshot.message)}</div>')
        if snapshot.summary:
            body.append(f'<div class="mc-lora-job-summary">{escape(snapshot.summary)}</div>')
        if snapshot.error:
            body.append(notice(escape(snapshot.error), kind="error"))
        if snapshot.log and (snapshot.active or snapshot.state == "failed"):
            tail = snapshot.log[-log_lines:]
            body.append(f'<pre class="mc-lora-log">{escape(chr(10).join(tail))}</pre>')
        blocks.append(f'<div class="mc-lora-job">{"".join(body)}</div>')

    return "".join(blocks)


def preset_folder_summary_html(preset: str, blocks) -> str:
    """What one UI Preset currently owns, per network type.

    ``blocks`` is ``[(page title, [(folder, includes subfolders, exists)])]``.
    """
    name = escape(preset or "—")
    assigned = [(title, rows) for title, rows in (blocks or []) if rows]
    if not assigned:
        return notice(
            f"Nothing is assigned to <b>{name}</b> yet — every network browser stays unfiltered "
            "while this preset is active.",
            kind="empty",
        )

    sections = [f'<div class="mc-lora-source-title">Assigned to <b>{name}</b></div>']
    for title, rows in assigned:
        body = "".join(
            "<tr>"
            f"<td><code>{escape(folder)}</code></td>"
            f"<td>{'this folder and its subfolders' if subfolders else 'this folder only'}</td>"
            f"<td>{'' if exists else 'folder is missing'}</td>"
            "</tr>"
            for folder, subfolders, exists in rows
        )
        sections.append(
            f'<div class="mc-lora-source">'
            f'<div class="mc-lora-source-title">{escape(title)} <span class="mc-lora-muted">· {len(rows)} folder(s)</span></div>'
            '<table class="mc-lora-table mc-lora-grid">'
            "<thead><tr><th>Folder</th><th>Scope</th><th>Problem</th></tr></thead>"
            f"<tbody>{body}</tbody></table>"
            "</div>"
        )
    return "".join(sections)
