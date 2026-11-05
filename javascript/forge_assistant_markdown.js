// Forge Assistant -- a message's words, drawn as HTML the panel may insert.
//
// Asked for: "Sometimes gemma4 responds with syntax like "$\rightarrow$", i
// want for all of the syntax formats to be handled correctly. Also, when text
// is in quotes like "this" i want it to be slightly more highlight. I think
// oobabooga does well with formatting." The panel used to draw bold, italics,
// inline code, fenced code, links and paragraphs; everything else a model
// writes -- headings, lists, tables, quotes, rules, strikethrough, math --
// reached the reader as the characters that were meant to make it.
//
// Two surfaces, one reading
// -------------------------
// LLM Studio's transcript is a Gradio Chatbot, whose Markdown is marked's and
// whose math is the KaTeX Gradio bundles; this panel draws its own. What the
// two must agree on is what a message says: where its math is, which words are
// quoted, which text is code and so never touched. The first half of this file
// is that reading, the same rules as mc_llm_format.py in the same order -- code
// lines, then the pieces nothing may reach into (code spans, math, tags, a
// link's address, escapes), then quotes -- and both are held to one table of
// cases, tests/fixtures/chat_format_cases.json. The math is lifted out before
// the Markdown is read and put back after, as Gradio does, so a formula's `*`,
// `_` and `\\` are never Markdown's.
//
// Safe by construction, as before
// -------------------------------
// The words are escaped and every element and attribute in the result is one
// this file writes: paragraphs, headings, lists, a table, block quotes, rules,
// code, emphasis, <q> for a quotation, a few inline tags a model writes bare
// (<br>, <sup>, <b>...: written again by this file, never copied, and never
// with an attribute), and links whose address passes `safeUrl` -- http, https,
// mailto or a path on this page -- opened in a new tab without an opener. A
// picture is a link to it: nothing in a reply makes the page fetch anything.
// The one HTML this file does not write is KaTeX's, for the math, made by the
// vendored KaTeX with `trust` off, which is its own guarantee: no \href, no
// \url, no \includegraphics, no class or style a formula asks for -- those are
// drawn in red as the TeX that asked.
//
// KaTeX, fetched when needed
// --------------------------
// Version 0.16.10, the one Gradio 4.40 bundles, so a formula looks the same in
// the panel and in the tab (vendor/katex, served by mc_katex.py). A page with
// no math in it fetches nothing. The first message with math draws each
// formula as it was written, in a quiet monospace, and asks for the script and
// the stylesheet; once both are in, every formula on screen is typeset where
// it stands, and every message after is drawn typeset straight away. A failed
// fetch leaves the formulas as written and is tried again after a minute.

(function () {
    "use strict";

    const NS = (window.forgeAssistant = window.forgeAssistant || {});

    // -- the reading (mc_llm_format.py) ------------------------------------- //

    const ENVIRONMENTS = new Set([
        "equation", "equation*", "align", "align*", "alignat", "alignat*", "gather", "gather*",
        "aligned", "alignedat", "gathered", "split", "CD",
        "matrix", "pmatrix", "bmatrix", "Bmatrix", "vmatrix", "Vmatrix",
        "matrix*", "pmatrix*", "bmatrix*", "Bmatrix*", "vmatrix*", "Vmatrix*", "smallmatrix",
        "cases", "dcases", "rcases", "drcases", "array", "darray", "subarray",
    ]);
    const QUOTE_PAIRS = {
        "\"": ["\""],
        "\u201c": ["\u201d"],
        "\u201e": ["\u201c", "\u201d"],
        "\u00ab": ["\u00bb"],
        "\u300c": ["\u300d"],
        "\u300e": ["\u300f"],
    };
    const ASCII_PUNCTUATION = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~";
    const CODE = "code";
    const MATH = "math";
    const VERBATIM = "verbatim";
    const ESCAPE = "escape";

    const FENCE = /^[ \t]*(`{3,}|~{3,})(.*)$/;
    const FENCE_CLOSE = /^[ \t]*(`{3,}|~{3,})[ \t]*$/;
    const LIST_ITEM = /^(?:[-*+]|\d{1,9}[.)])(?:[ \t]|$)/;
    const BLANK = /^[ \t]*$/;
    const BLANK_LINE = /\n[ \t]*\n/g;
    const ENVIRONMENT = /\\begin\{([A-Za-z]+\*?)\}/y;
    const TAG = new RegExp(
        "<(?:[A-Za-z][A-Za-z0-9-]*(?:\\s+[A-Za-z_:][A-Za-z0-9_.:-]*"
        + "(?:\\s*=\\s*(?:[^\\s\"'=<>`]+|'[^'\\n]*'|\"[^\"\\n]*\"))?)*\\s*/?"
        + "|/[A-Za-z][A-Za-z0-9-]*\\s*)>", "y");
    const AUTOLINK = new RegExp(
        "<(?:[A-Za-z][A-Za-z0-9+.-]{1,31}:[^\\s<>]*"
        + "|[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+@[A-Za-z0-9](?:[A-Za-z0-9.-]*[A-Za-z0-9])?)>", "y");
    const ATOM_START = /[\\`$<\]]/g;
    const QUOTE_START = /[~"\u201c\u201e\u00ab\u300c\u300e\\]/g;
    const SPACE = /\s/;
    const LETTER_OR_DIGIT = /[\p{L}\p{N}]/u;

    function isSpace(char) {
        return !!char && SPACE.test(char);
    }

    function indentWidth(line) {
        let width = 0;
        for (let index = 0; index < line.length; index += 1) {
            const char = line[index];
            if (char === " ") width += 1;
            else if (char === "\t") width += 4 - (width % 4);
            else break;
        }
        return width;
    }

    function indented(line) {
        return indentWidth(line) >= 4;
    }

    /** Which of `lines` are code: fenced blocks at any indentation (one never
     *  closed runs to the end) and blocks indented four spaces after a blank
     *  line below flush prose that is not a list item. */
    function codeLines(lines) {
        const code = lines.map(() => false);
        let previous = null;
        let afterBlank = true;
        let index = 0;
        while (index < lines.length) {
            const line = lines[index];
            const opened = FENCE.exec(line);
            if (opened && !(opened[1][0] === "`" && opened[2].indexOf("`") >= 0)) {
                const char = opened[1][0];
                const size = opened[1].length;
                code[index] = true;
                index += 1;
                while (index < lines.length) {
                    code[index] = true;
                    const closed = FENCE_CLOSE.exec(lines[index]);
                    index += 1;
                    if (closed && closed[1][0] === char && closed[1].length >= size) break;
                }
                previous = lines[index - 1];
                afterBlank = false;
                continue;
            }
            if (BLANK.test(line)) {
                afterBlank = true;
                index += 1;
                continue;
            }
            if (afterBlank && indented(line)
                    && (previous === null || (!/^[ \t]/.test(previous) && !LIST_ITEM.test(previous)))) {
                let last = index;
                let ahead = index;
                while (ahead < lines.length && (BLANK.test(lines[ahead]) || indented(lines[ahead]))) {
                    if (!BLANK.test(lines[ahead])) last = ahead;
                    ahead += 1;
                }
                for (let covered = index; covered <= last; covered += 1) code[covered] = true;
                previous = lines[last];
                afterBlank = false;
                index = last + 1;
                continue;
            }
            previous = line;
            afterBlank = false;
            index += 1;
        }
        return code;
    }

    function escapedAt(text, index) {
        let count = 0;
        while (index - 1 - count >= 0 && text[index - 1 - count] === "\\") count += 1;
        return count % 2 === 1;
    }

    /** Where the first blank line from `start` begins; the end of the text if none. */
    function blankFrom(text, start) {
        BLANK_LINE.lastIndex = start;
        const found = BLANK_LINE.exec(text);
        return found ? found.index : text.length;
    }

    /** An unescaped `closer` from `start`, before a blank line; -1 if nowhere. */
    function closing(text, start, closer) {
        const end = blankFrom(text, start);
        let found = text.indexOf(closer, start);
        while (found >= 0 && found + closer.length <= end && escapedAt(text, found)) {
            found = text.indexOf(closer, found + 1);
        }
        return found >= 0 && found + closer.length <= end ? found : -1;
    }

    /** The end of a run of exactly `size` backticks after `from`, before
     *  `end`; -1 if there is none. */
    function backtickRun(text, from, end, size) {
        let at = text.indexOf("`", from);
        while (at >= 0 && at < end) {
            let stop = at;
            while (stop < text.length && text[stop] === "`") stop += 1;
            if (stop - at === size && stop <= end) return stop;
            at = text.indexOf("`", stop);
        }
        return -1;
    }

    function environmentEnd(text, index) {
        ENVIRONMENT.lastIndex = index;
        const begun = ENVIRONMENT.exec(text);
        if (!begun || !ENVIRONMENTS.has(begun[1])) return -1;
        const name = begun[1].replace(/\*/g, "\\*");
        const marks = new RegExp("\\\\(begin|end)\\{" + name + "\\}", "g");
        const start = index + begun[0].length;
        const end = blankFrom(text, start);
        marks.lastIndex = start;
        let depth = 1;
        let found = marks.exec(text);
        while (found && found.index + found[0].length <= end) {
            depth += found[1] === "begin" ? 1 : -1;
            if (depth === 0) return found.index + found[0].length;
            found = marks.exec(text);
        }
        return -1;
    }

    /** The end of `$…$` math opened at `index`, past its closing `$`; -1 if
     *  it is not math (Pandoc's rule, the first `$` after the opening one). */
    function inlineDollarEnd(text, index) {
        const after = index + 1;
        if (after >= text.length || isSpace(text[after]) || text[after] === "$") return -1;
        let position = after;
        while (position < text.length) {
            const char = text[position];
            if (char === "\\") {
                position += 2;
                continue;
            }
            if (char === "\n" || char === "`") return -1;
            if (char === "$") {
                if (isSpace(text[position - 1])) return -1;
                const following = position + 1 < text.length ? text[position + 1] : "";
                if (following >= "0" && following <= "9") return -1;
                return position + 1;
            }
            position += 1;
        }
        return -1;
    }

    /** The end of a link's `(address "title")` after the `]` at `index`; -1 if none. */
    function destinationEnd(text, index) {
        if (text[index] !== "]" || text[index + 1] !== "(") return -1;
        let depth = 0;
        let position = index + 2;
        while (position < text.length) {
            const char = text[position];
            if (char === "\n") return -1;
            if (char === "\\") {
                position += 2;
                continue;
            }
            if (char === "(") {
                depth += 1;
            } else if (char === ")") {
                if (depth === 0) return position + 1;
                depth -= 1;
            }
            position += 1;
        }
        return -1;
    }

    /** The pieces of prose that nothing may reach into: [start, end, kind,
     *  payload], in order -- code spans, math ([tex, display]), tags,
     *  autolinks and links' addresses, escapes. */
    function atoms(text) {
        const found = [];
        const length = text.length;
        let index = 0;
        for (;;) {
            ATOM_START.lastIndex = index;
            const hit = ATOM_START.exec(text);
            if (!hit) break;
            index = hit.index;
            const char = text[index];
            if (char === "\\") {
                if (text.startsWith("\\(", index)) {
                    const close = closing(text, index + 2, "\\)");
                    if (close >= 0 && text.slice(index + 2, close).trim()) {
                        found.push([index, close + 2, MATH, [text.slice(index + 2, close), false]]);
                        index = close + 2;
                        continue;
                    }
                } else if (text.startsWith("\\[", index)) {
                    const close = closing(text, index + 2, "\\]");
                    if (close >= 0 && text.slice(index + 2, close).trim()) {
                        found.push([index, close + 2, MATH, [text.slice(index + 2, close), true]]);
                        index = close + 2;
                        continue;
                    }
                } else if (text.startsWith("\\begin{", index)) {
                    const end = environmentEnd(text, index);
                    if (end >= 0) {
                        found.push([index, end, MATH, [text.slice(index, end), true]]);
                        index = end;
                        continue;
                    }
                }
                if (index + 1 < length && ASCII_PUNCTUATION.indexOf(text[index + 1]) >= 0) {
                    found.push([index, index + 2, ESCAPE, null]);
                    index += 2;
                } else {
                    index += 1;
                }
                continue;
            }
            if (char === "`") {
                let run = index;
                while (run < length && text[run] === "`") run += 1;
                const stop = backtickRun(text, run, blankFrom(text, run), run - index);
                if (stop >= 0) {
                    found.push([index, stop, CODE, null]);
                    index = stop;
                } else {
                    index = run;
                }
                continue;
            }
            if (char === "$") {
                let run = index;
                while (run < length && text[run] === "$") run += 1;
                const size = run - index;
                if (size === 2 && !escapedAt(text, index)) {
                    const close = closing(text, run, "$$");
                    if (close >= 0 && text.slice(run, close).trim()) {
                        found.push([index, close + 2, MATH, [text.slice(run, close), true]]);
                        index = close + 2;
                        continue;
                    }
                } else if (size === 1 && !escapedAt(text, index)) {
                    const end = inlineDollarEnd(text, index);
                    if (end >= 0) {
                        found.push([index, end, MATH, [text.slice(index + 1, end - 1), false]]);
                        index = end;
                        continue;
                    }
                }
                index = run;
                continue;
            }
            if (char === "<") {
                TAG.lastIndex = index;
                AUTOLINK.lastIndex = index;
                const tag = TAG.exec(text) || AUTOLINK.exec(text);
                if (tag) {
                    found.push([index, index + tag[0].length, VERBATIM, null]);
                    index += tag[0].length;
                    continue;
                }
                index += 1;
                continue;
            }
            const end = destinationEnd(text, index);   // char === "]"
            if (end >= 0) {
                found.push([index + 1, end, VERBATIM, null]);
                index = end;
                continue;
            }
            index += 1;
        }
        return found;
    }

    function letterOrDigit(char) {
        return !!char && LETTER_OR_DIGIT.test(char);
    }

    function opens(text, index) {
        const mark = text[index];
        const after = index + 1 < text.length ? text[index + 1] : "";
        if (!after) return false;
        if (mark === "\u00ab") return true;
        if (isSpace(after)) return false;
        if (mark === "\"") {
            const before = index > 0 ? text[index - 1] : "";
            return after !== "\"" && !letterOrDigit(before);
        }
        return true;
    }

    function closes(text, index, opener) {
        if (opener === "\u00ab") return true;
        const before = index > 0 ? text[index - 1] : "";
        if (!before || isSpace(before)) return false;
        if (opener === "\"") {
            const after = index + 1 < text.length ? text[index + 1] : "";
            return !letterOrDigit(after);
        }
        return true;
    }

    /** The quotes in prose, as [opening index, closing index] of their marks,
     *  and its lone tildes, given its atoms. */
    function marks(text, found) {
        const quotes = [];
        const tildes = [];
        const starts = new Map(found.map((atom) => [atom[0], atom]));
        const length = text.length;
        let index = 0;
        let following = 0;
        while (index < length) {
            while (following < found.length && found[following][1] <= index) following += 1;
            if (following < found.length && found[following][0] <= index) {
                index = found[following][1];
                continue;
            }
            const bound = following < found.length ? found[following][0] : length;
            QUOTE_START.lastIndex = index;
            const hit = QUOTE_START.exec(text);
            if (!hit || hit.index >= bound) {
                index = bound;
                continue;
            }
            index = hit.index;
            const char = text[index];
            if (char === "\\") {
                index += 1;
                continue;
            }
            if (char === "~") {
                let run = index;
                while (run < bound && text[run] === "~") run += 1;
                if (run - index === 1) tildes.push(index);
                index = run;
                continue;
            }
            if (!opens(text, index)) {
                index += 1;
                continue;
            }
            const closers = QUOTE_PAIRS[char];
            let position = index + 1;
            let closed = -1;
            while (position < length) {
                const atom = starts.get(position);
                if (atom) {
                    if (atom[2] === MATH && atom[3][1]) break;
                    position = atom[1];
                    continue;
                }
                const here = text[position];
                if (here === "\n" || here === "|") break;
                if (here === "\\") {
                    position += 1;
                    continue;
                }
                if (closers.indexOf(here) >= 0 && closes(text, position, char)) {
                    closed = position;
                    break;
                }
                if (here === char && opens(text, position)) break;   // another quote opens
                position += 1;
            }
            if (closed > 0 && hasWords(text, index + 1, closed, starts)) {
                quotes.push([index, closed]);
                index = closed + 1;
            } else {
                index += 1;
            }
        }
        return [quotes, tildes];
    }

    function hasWords(text, start, end, starts) {
        for (let at = start; at < end; at += 1) {
            if (!isSpace(text[at]) || starts.has(at)) return true;
        }
        return false;
    }

    function normalised(text) {
        return String(text === undefined || text === null ? "" : text)
            .replace(/\r\n/g, "\n").replace(/\r/g, "\n");
    }

    /** The message as [isCode, text] runs of whole lines. */
    function chunks(text) {
        const lines = text.split("\n");
        const flags = codeLines(lines);
        const runs = [];
        lines.forEach((line, index) => {
            const last = runs[runs.length - 1];
            if (last && last[0] === flags[index]) last[1].push(line);
            else runs.push([flags[index], [line]]);
        });
        return runs.map((run) => [run[0], run[1].join("\n")]);
    }

    /** What the rules find in a message, in the shape mc_llm_format.describe
     *  answers in: {math: [[tex, display], ...], quotes: [text, ...]}. */
    function describeMessage(text) {
        const math = [];
        const quotes = [];
        chunks(normalised(text)).forEach((run) => {
            if (run[0]) return;
            const body = run[1];
            const found = atoms(body);
            found.forEach((atom) => {
                if (atom[2] === MATH) math.push([atom[3][0], atom[3][1]]);
            });
            marks(body, found)[0].forEach((pair) => quotes.push(body.slice(pair[0], pair[1] + 1)));
        });
        return {math, quotes};
    }

    // -- KaTeX, fetched when first needed --------------------------------- //

    const KATEX_VERSION = "0.16.10";
    const KATEX_PATH = "/model-chain/katex/" + KATEX_VERSION + "/";
    const KATEX_RETRY_MS = 60000;
    const KATEX_CACHE_LIMIT = 400;
    // `trust` off is the guarantee above; `maxSize` keeps a \rule or a \kern
    // from making a bubble kilometres wide; `strict` "ignore" keeps a formula
    // KaTeX would warn about (Unicode in math, mostly) off the console.
    const KATEX_OPTIONS = {throwOnError: false, strict: "ignore", trust: false,
                           maxSize: 20, maxExpand: 1000, output: "htmlAndMathml"};
    const katex = {lib: null, loading: null, failedAt: 0, drawn: new Map()};

    function katexBase() {
        return (typeof NS.basePath === "function" ? NS.basePath() : "") + KATEX_PATH;
    }

    function typesetHtml(tex, display) {
        const key = (display ? "D" : "I") + tex;
        const held = katex.drawn.get(key);
        if (held !== undefined) return held;
        let made;
        try {
            made = katex.lib.renderToString(tex, Object.assign({displayMode: display}, KATEX_OPTIONS));
        } catch (error) {
            made = null;
        }
        if (katex.drawn.size >= KATEX_CACHE_LIMIT) katex.drawn.clear();
        katex.drawn.set(key, made);
        return made;
    }

    /** Every formula still waiting on screen, typeset where it stands. */
    function typesetPending(root) {
        if (!katex.lib || typeof document === "undefined") return 0;
        const scope = root || document;
        if (!scope.querySelectorAll) return 0;
        const waiting = scope.querySelectorAll(".forge-assistant-math[data-tex]");
        let done = 0;
        for (let index = 0; index < waiting.length; index += 1) {
            const node = waiting[index];
            const made = typesetHtml(node.getAttribute("data-tex") || "",
                                     node.getAttribute("data-display") === "1");
            if (made === null) continue;
            node.outerHTML = made;
            done += 1;
        }
        return done;
    }

    function loadKatex() {
        if (katex.lib || katex.loading) return katex.loading;
        if (katex.failedAt && Date.now() - katex.failedAt < KATEX_RETRY_MS) return null;
        if (typeof fetch !== "function" || typeof document === "undefined" || !document.head) {
            return null;
        }
        const base = katexBase();
        const stylesheet = new Promise((resolve, reject) => {
            const kept = document.querySelector("link[data-forge-assistant-katex]");
            if (kept) {
                resolve();
                return;
            }
            const link = document.createElement("link");
            link.rel = "stylesheet";
            link.href = base + "katex.min.css";
            link.setAttribute("data-forge-assistant-katex", KATEX_VERSION);
            link.onload = () => resolve();
            link.onerror = () => {
                if (link.parentNode) link.parentNode.removeChild(link);
                reject(new Error("the stylesheet did not load"));
            };
            document.head.appendChild(link);
        });
        // Fetched and run with a module of its own rather than a <script>:
        // the bundle is UMD, and on a page with an AMD loader it registers
        // with that loader instead of the window and is never found.
        const script = fetch(base + "katex.min.js", {credentials: "same-origin"})
            .then((response) => {
                if (!response.ok) throw new Error("HTTP " + response.status);
                return response.text();
            })
            .then((source) => {
                const holder = {exports: {}};
                // eslint-disable-next-line no-new-func
                new Function("module", "exports", "define", source)(holder, holder.exports, undefined);
                if (!holder.exports || typeof holder.exports.renderToString !== "function") {
                    throw new Error("the script is not KaTeX");
                }
                return holder.exports;
            });
        katex.loading = Promise.all([script, stylesheet]).then((loaded) => {
            katex.lib = loaded[0];
            katex.loading = null;
            typesetPending();
            return katex.lib;
        }, (error) => {
            katex.loading = null;
            katex.failedAt = Date.now();
            console.warn("Forge Assistant: math is shown as written; KaTeX did not load", error);
            return null;
        });
        return katex.loading;
    }

    /** A loaded KaTeX, handed in (the tests' way in, and a page's that has one). */
    function useKatex(lib) {
        katex.lib = lib && typeof lib.renderToString === "function" ? lib : null;
        katex.drawn.clear();
        return !!katex.lib;
    }

    // -- the renderer ------------------------------------------------------ //

    // Characters of the private use area this file uses for itself while it
    // draws: a formula, a quote's two ends, a piece already drawn. A message
    // that contains one has it replaced before anything is read.
    const MATH_OPEN = "\uE000";
    const MATH_CLOSE = "\uE001";
    const QUOTE_OPEN = "\uE002";
    const QUOTE_CLOSE = "\uE003";
    const OURS = /[\u0000\uE000-\uE00F]/g;
    const MATH_TOKEN = /\uE000(\d+)\uE001/g;

    function escapeHtml(text) {
        return String(text === undefined || text === null ? "" : text)
            .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;").replace(/'/g, "&#39;");
    }

    function safeUrl(raw) {
        const url = String(raw || "").trim();
        // Nothing invisible in an address: a control character, or one of the
        // marks this file draws with.
        if (/[\u0000-\u001f\u007f\uE000-\uE00F]/.test(url)) return "";
        // An allow-list of three schemes and nothing else. `javascript:`,
        // `data:` and `vbscript:` are the obvious ones; the list is positive
        // rather than negative because the next scheme somebody finds is one a
        // blocklist has never heard of.
        if (/^(https?:|mailto:)/i.test(url)) return url;
        if (/^\//.test(url) && !/^\/\//.test(url)) return url;   // same-origin path
        return "";
    }

    function linkOpen(url) {
        // A new tab without an opener: a link in a reply must not be able to
        // reach back into this page.
        return '<a href="' + escapeHtml(url) + '" target="_blank" rel="noopener noreferrer">';
    }

    /** Text with the drawing's own marks taken out again: a formula as it was
     *  written, a quote without its ends. For code, where nothing is drawn. */
    function asWritten(text, context) {
        return String(text).replace(MATH_TOKEN, (match, index) => context.maths[Number(index)].source)
            .replace(/[\uE002\uE003]/g, "");
    }

    function mathHtml(context, index) {
        const math = context.maths[index];
        if (!math) return "";
        if (katex.lib) {
            const made = typesetHtml(math.tex, math.display);
            if (made !== null) return made;
        }
        context.wantsKatex = true;
        return '<span class="forge-assistant-math' + (math.display ? " forge-assistant-math-display" : "")
            + '" data-tex="' + escapeHtml(math.tex) + '" data-display="' + (math.display ? "1" : "0")
            + '">' + escapeHtml(math.source) + "</span>";
    }

    // -- blocks --

    const ATX = /^ {0,3}(#{1,6})(?:[ \t]+(.*?))?(?:[ \t]+#+)?[ \t]*$/;
    const HR = /^ {0,3}(?:(?:\*[ \t]*){3,}|(?:-[ \t]*){3,}|(?:_[ \t]*){3,})$/;
    const SETEXT = /^ {0,3}(=+|-+)[ \t]*$/;
    const QUOTE_LINE = /^ {0,3}> ?/;
    const MARKER = /^( {0,3})([-*+]|\d{1,9}[.)])(?=[ \t]|$)/;
    const DELIMITER_CELL = /^:?-+:?$/;
    const TASK = /^\[([ xX])\](?:[ \t]+|$)/;
    const DISPLAY_ALONE = /^\uE000(\d+)\uE001$/;

    function expandIndent(line) {
        const lead = /^[ \t]+/.exec(line);
        if (!lead || lead[0].indexOf("\t") < 0) return line;
        return " ".repeat(indentWidth(lead[0])) + line.slice(lead[0].length);
    }

    function stripIndent(line, width) {
        let removed = 0;
        let index = 0;
        while (index < line.length && removed < width && line[index] === " ") {
            removed += 1;
            index += 1;
        }
        return line.slice(index);
    }

    function listMarker(line) {
        const found = MARKER.exec(line);
        if (!found || HR.test(line)) return null;
        const indent = found[1].length;
        const marker = found[2];
        const ordered = marker[0] >= "0" && marker[0] <= "9";
        const rest = line.slice(found[0].length);
        let spacing = 0;
        while (spacing < rest.length && rest[spacing] === " ") spacing += 1;
        let text = rest.slice(spacing);
        if (!text.trim()) {
            spacing = 1;
            text = "";
        } else if (spacing > 4) {
            text = rest.slice(1);
            spacing = 1;
        }
        return {
            indent, ordered, text,
            kind: ordered ? marker.slice(-1) : marker,
            start: ordered ? parseInt(marker, 10) : 1,
            offset: indent + marker.length + Math.max(1, spacing),
        };
    }

    function splitRow(line) {
        let row = line.trim();
        if (row[0] === "|") row = row.slice(1);
        if (row[row.length - 1] === "|" && row[row.length - 2] !== "\\") row = row.slice(0, -1);
        const cells = [];
        let cell = "";
        for (let index = 0; index < row.length; index += 1) {
            const char = row[index];
            if (char === "\\" && row[index + 1] === "|") {
                cell += "|";
                index += 1;
                continue;
            }
            if (char === "|") {
                cells.push(cell.trim());
                cell = "";
                continue;
            }
            cell += char;
        }
        cells.push(cell.trim());
        return cells;
    }

    /** A table's header and delimiter rows at `index`, as their cells and
     *  alignments, or null. */
    function tableAt(lines, index) {
        if (index + 1 >= lines.length) return null;
        const head = lines[index];
        const rule = lines[index + 1];
        if (head.indexOf("|") < 0 || rule.indexOf("|") < 0 || indented(head)) return null;
        const cells = splitRow(head);
        const rules = splitRow(rule);
        if (rules.length !== cells.length || !rules.every((cell) => DELIMITER_CELL.test(cell))) {
            return null;
        }
        const aligns = rules.map((cell) => {
            const left = cell[0] === ":";
            const right = cell[cell.length - 1] === ":";
            return left && right ? "center" : (right ? "right" : (left ? "left" : ""));
        });
        return {cells, aligns};
    }

    function startsBlock(line) {
        return FENCE.test(line) || ATX.test(line) || HR.test(line) || QUOTE_LINE.test(line)
            || !!listMarker(line);
    }

    /** Whether `line` ends the paragraph above it (CommonMark's rules, with
     *  marked's table): a list that interrupts one has words and, numbered,
     *  starts at 1. */
    function interrupts(lines, index) {
        const line = lines[index];
        if (FENCE.test(line) || ATX.test(line) || HR.test(line) || QUOTE_LINE.test(line)) return true;
        const item = listMarker(line);
        if (item && item.text && (!item.ordered || item.start === 1)) return true;
        return !!tableAt(lines, index);
    }

    function renderBlocks(source, context, tight) {
        const lines = source.map(expandIndent);
        let html = "";
        let previous = null;
        let afterBlank = true;
        let index = 0;
        while (index < lines.length) {
            const line = lines[index];
            if (BLANK.test(line)) {
                afterBlank = true;
                index += 1;
                continue;
            }
            const fence = FENCE.exec(line);
            if (fence && !(fence[1][0] === "`" && fence[2].indexOf("`") >= 0)) {
                const char = fence[1][0];
                const size = fence[1].length;
                const lead = indentWidth(line);
                const body = [];
                index += 1;
                while (index < lines.length) {
                    const closed = FENCE_CLOSE.exec(lines[index]);
                    if (closed && closed[1][0] === char && closed[1].length >= size) {
                        index += 1;
                        break;
                    }
                    body.push(stripIndent(lines[index], lead));
                    index += 1;
                }
                html += codeBlock(body.join("\n"), context);
                previous = lines[index - 1];
                afterBlank = false;
                continue;
            }
            if (afterBlank && indented(line)
                    && (previous === null || (!/^[ \t]/.test(previous) && !LIST_ITEM.test(previous)))) {
                let last = index;
                let ahead = index;
                while (ahead < lines.length && (BLANK.test(lines[ahead]) || indented(lines[ahead]))) {
                    if (!BLANK.test(lines[ahead])) last = ahead;
                    ahead += 1;
                }
                html += codeBlock(lines.slice(index, last + 1).map((held) => stripIndent(held, 4))
                    .join("\n"), context);
                previous = lines[last];
                afterBlank = false;
                index = last + 1;
                continue;
            }
            previous = line;
            afterBlank = false;
            const heading = ATX.exec(line);
            if (heading) {
                const level = heading[1].length;
                html += "<h" + level + ">" + inline((heading[2] || "").trim(), context) + "</h" + level + ">";
                index += 1;
                continue;
            }
            if (HR.test(line)) {
                html += "<hr>";
                index += 1;
                continue;
            }
            if (QUOTE_LINE.test(line)) {
                const inner = [];
                while (index < lines.length) {
                    const here = lines[index];
                    if (QUOTE_LINE.test(here)) {
                        inner.push(here.replace(QUOTE_LINE, ""));
                    } else if (!BLANK.test(here) && !startsBlock(here) && inner.length
                               && !BLANK.test(inner[inner.length - 1])) {
                        inner.push(here);   // a lazy continuation of its paragraph
                    } else {
                        break;
                    }
                    index += 1;
                }
                html += "<blockquote>" + renderBlocks(inner, context, false) + "</blockquote>";
                previous = lines[index - 1];
                continue;
            }
            if (listMarker(line)) {
                const list = parseList(lines, index);
                html += list.html(context);
                index = list.end;
                previous = lines[index - 1];
                continue;
            }
            const table = tableAt(lines, index);
            if (table) {
                index += 2;
                const rows = [];
                while (index < lines.length && !BLANK.test(lines[index]) && lines[index].indexOf("|") >= 0
                       && !(FENCE.test(lines[index]) || ATX.test(lines[index]) || QUOTE_LINE.test(lines[index]))) {
                    rows.push(splitRow(lines[index]));
                    index += 1;
                }
                html += tableHtml(table, rows, context);
                previous = lines[index - 1];
                continue;
            }
            // A paragraph, or a setext heading.
            const words = [line.trim()];
            index += 1;
            let level = 0;
            while (index < lines.length && !BLANK.test(lines[index])) {
                const setext = SETEXT.exec(lines[index]);
                if (setext) {
                    level = setext[1][0] === "=" ? 1 : 2;
                    index += 1;
                    break;
                }
                if (interrupts(lines, index)) break;
                words.push(lines[index].trim());
                index += 1;
            }
            previous = lines[index - 1];
            const text = words.join("\n");
            if (level) {
                html += "<h" + level + ">" + inline(text, context) + "</h" + level + ">";
                continue;
            }
            const alone = DISPLAY_ALONE.exec(text);
            if (alone && context.maths[Number(alone[1])] && context.maths[Number(alone[1])].display) {
                html += mathHtml(context, Number(alone[1]));
            } else if (tight) {
                html += inline(text, context);
            } else {
                html += "<p>" + inline(text, context) + "</p>";
            }
        }
        return html;
    }

    function codeBlock(text, context) {
        return '<pre class="forge-assistant-code"><code>' + escapeHtml(asWritten(text, context))
            + "</code></pre>";
    }

    function tableHtml(table, rows, context) {
        const cell = (tag, text, align) => "<" + tag
            + (align ? ' class="forge-assistant-align-' + align + '"' : "") + ">"
            + inline(text, context) + "</" + tag + ">";
        let html = "<table><thead><tr>"
            + table.cells.map((text, column) => cell("th", text, table.aligns[column])).join("")
            + "</tr></thead>";
        if (rows.length) {
            html += "<tbody>" + rows.map((row) => "<tr>" + table.cells.map((unused, column) =>
                cell("td", row[column] || "", table.aligns[column])).join("") + "</tr>").join("")
                + "</tbody>";
        }
        return html + "</table>";
    }

    /** A list from `start`: its items, each the lines that belong to it with
     *  the item's indentation taken off, and whether it is loose. */
    function parseList(lines, start) {
        const first = listMarker(lines[start]);
        const items = [];
        let index = start;
        let loose = false;
        while (index < lines.length) {
            const item = listMarker(lines[index]);
            if (!item || item.ordered !== first.ordered || item.kind !== first.kind) break;
            const body = [item.text];
            index += 1;
            let sawBlank = false;
            while (index < lines.length) {
                const line = lines[index];
                if (BLANK.test(line)) {
                    body.push("");
                    sawBlank = true;
                    index += 1;
                    continue;
                }
                if (indentWidth(line) >= item.offset) {
                    body.push(stripIndent(line, item.offset));
                    sawBlank = false;
                    index += 1;
                    continue;
                }
                if (sawBlank || startsBlock(line)) break;
                body.push(line.trim());   // a lazy continuation of the item's paragraph
                index += 1;
            }
            let trailing = 0;
            while (body.length > 1 && BLANK.test(body[body.length - 1])) {
                body.pop();
                trailing += 1;
            }
            if (trailing && index < lines.length) {
                const next = listMarker(lines[index]);
                if (next && next.ordered === first.ordered && next.kind === first.kind) loose = true;
            }
            for (let at = 1; at < body.length - 1; at += 1) {
                if (BLANK.test(body[at]) && !BLANK.test(body[at - 1]) && !BLANK.test(body[at + 1])
                    && !listMarker(body[at + 1])) {
                    loose = true;
                }
            }
            items.push(body);
        }
        return {
            end: index,
            html(context) {
                const tag = first.ordered ? "ol" : "ul";
                const opening = first.ordered && first.start !== 1
                    ? '<ol start="' + first.start + '">' : "<" + tag + ">";
                return opening + items.map((body) => {
                    const task = TASK.exec(body[0]);
                    if (task) {
                        const done = task[1] !== " ";
                        body[0] = body[0].slice(task[0].length);
                        return '<li class="forge-assistant-task"><span class="forge-assistant-check" aria-hidden="true">'
                            + (done ? "\u2611" : "\u2610") + "</span> " + renderBlocks(body, context, !loose) + "</li>";
                    }
                    return "<li>" + renderBlocks(body, context, !loose) + "</li>";
                }).join("") + "</" + tag + ">";
            },
        };
    }

    // -- inlines --

    const TAG_NAMES = new Set(["b", "strong", "i", "em", "u", "s", "del", "ins", "mark", "sub",
                               "sup", "small", "kbd", "code"]);
    const INLINE_TAG = /<(\/?)([A-Za-z]+)\s*>/y;
    const BREAK_TAG = /<br\s*\/?>/iy;
    const ENTITY = /&(?:#(\d{1,7})|#[xX]([0-9a-fA-F]{1,6})|([A-Za-z][A-Za-z0-9]{1,31}));/y;
    // Up to a space, a `<` or one of this file's own marks: an address never
    // runs on into a formula or a quote drawn after it.
    const BARE_URL = /(?:https?:\/\/|www\.)[^\s<\uE000-\uE00F]*/iy;
    const PUNCTUATION = /[\p{P}\p{S}]/u;
    const ENTITIES = {
        amp: "&", lt: "<", gt: ">", quot: "\"", apos: "'", nbsp: "\u00a0", ensp: "\u2002",
        emsp: "\u2003", thinsp: "\u2009", zwj: "\u200d", zwnj: "\u200c", shy: "\u00ad",
        hellip: "\u2026", mdash: "\u2014", ndash: "\u2013", lsquo: "\u2018", rsquo: "\u2019",
        sbquo: "\u201a", ldquo: "\u201c", rdquo: "\u201d", bdquo: "\u201e", laquo: "\u00ab",
        raquo: "\u00bb", lsaquo: "\u2039", rsaquo: "\u203a", bull: "\u2022", middot: "\u00b7",
        deg: "\u00b0", plusmn: "\u00b1", times: "\u00d7", divide: "\u00f7", minus: "\u2212",
        le: "\u2264", ge: "\u2265", ne: "\u2260", asymp: "\u2248", equiv: "\u2261",
        infin: "\u221e", sum: "\u2211", prod: "\u220f", radic: "\u221a", int: "\u222b",
        part: "\u2202", nabla: "\u2207", isin: "\u2208", notin: "\u2209", sub: "\u2282",
        sube: "\u2286", cap: "\u2229", cup: "\u222a", and: "\u2227", or: "\u2228",
        not: "\u00ac", forall: "\u2200", exist: "\u2203", empty: "\u2205", prop: "\u221d",
        larr: "\u2190", rarr: "\u2192", uarr: "\u2191", darr: "\u2193", harr: "\u2194",
        lArr: "\u21d0", rArr: "\u21d2", uArr: "\u21d1", dArr: "\u21d3", hArr: "\u21d4",
        crarr: "\u21b5", micro: "\u00b5", para: "\u00b6", sect: "\u00a7", copy: "\u00a9",
        reg: "\u00ae", trade: "\u2122", euro: "\u20ac", pound: "\u00a3", yen: "\u00a5",
        cent: "\u00a2", curren: "\u00a4", frac12: "\u00bd", frac14: "\u00bc", frac34: "\u00be",
        sup1: "\u00b9", sup2: "\u00b2", sup3: "\u00b3", prime: "\u2032", Prime: "\u2033",
        dagger: "\u2020", Dagger: "\u2021", permil: "\u2030", star: "\u2606", starf: "\u2605",
        check: "\u2713", cross: "\u2717", hearts: "\u2665", spades: "\u2660", clubs: "\u2663",
        diams: "\u2666", loz: "\u25ca", iexcl: "\u00a1", iquest: "\u00bf", ordf: "\u00aa",
        ordm: "\u00ba", alpha: "\u03b1", beta: "\u03b2", gamma: "\u03b3", delta: "\u03b4",
        epsilon: "\u03b5", zeta: "\u03b6", eta: "\u03b7", theta: "\u03b8", iota: "\u03b9",
        kappa: "\u03ba", lambda: "\u03bb", mu: "\u03bc", nu: "\u03bd", xi: "\u03be",
        omicron: "\u03bf", pi: "\u03c0", rho: "\u03c1", sigma: "\u03c3", sigmaf: "\u03c2",
        tau: "\u03c4", upsilon: "\u03c5", phi: "\u03c6", chi: "\u03c7", psi: "\u03c8",
        omega: "\u03c9", Alpha: "\u0391", Beta: "\u0392", Gamma: "\u0393", Delta: "\u0394",
        Epsilon: "\u0395", Zeta: "\u0396", Eta: "\u0397", Theta: "\u0398", Iota: "\u0399",
        Kappa: "\u039a", Lambda: "\u039b", Mu: "\u039c", Nu: "\u039d", Xi: "\u039e",
        Omicron: "\u039f", Pi: "\u03a0", Rho: "\u03a1", Sigma: "\u03a3", Tau: "\u03a4",
        Upsilon: "\u03a5", Phi: "\u03a6", Chi: "\u03a7", Psi: "\u03a8", Omega: "\u03a9",
    };

    function entityAt(text, index) {
        ENTITY.lastIndex = index;
        const found = ENTITY.exec(text);
        if (!found) return null;
        let char = null;
        if (found[1] || found[2]) {
            const code = parseInt(found[1] || found[2], found[1] ? 10 : 16);
            char = code > 0 && code <= 0x10ffff && !(code >= 0xd800 && code <= 0xdfff)
                ? String.fromCodePoint(code) : "\ufffd";
        } else if (Object.prototype.hasOwnProperty.call(ENTITIES, found[3])) {
            char = ENTITIES[found[3]];
        }
        return char === null ? null : {char, end: index + found[0].length};
    }

    /** `[label](address "title")` from the `[` at `index`: the label and the
     *  address, and where it ends; null where it is not a link. */
    function linkAt(text, index) {
        let depth = 0;
        let position = index;
        let close = -1;
        while (position < text.length) {
            const char = text[position];
            if (char === "\\") {
                position += 2;
                continue;
            }
            if (char === "`") {
                let run = position;
                while (run < text.length && text[run] === "`") run += 1;
                const stop = backtickRun(text, run, text.length, run - position);
                position = stop >= 0 ? stop : run;
                continue;
            }
            if (char === "[") depth += 1;
            if (char === "]") {
                depth -= 1;
                if (depth === 0) {
                    close = position;
                    break;
                }
            }
            position += 1;
        }
        if (close < 0 || text[close + 1] !== "(") return null;
        const end = destinationEnd(text, close);
        if (end < 0) return null;
        const inside = text.slice(close + 2, end - 1).trim();
        let address;
        let rest;
        if (inside[0] === "<") {
            const shut = inside.indexOf(">");
            if (shut < 0) return null;
            address = inside.slice(1, shut);
            rest = inside.slice(shut + 1).trim();
        } else {
            const space = inside.search(/\s/);
            address = space < 0 ? inside : inside.slice(0, space);
            rest = space < 0 ? "" : inside.slice(space).trim();
        }
        if (rest && !/^(?:"[^"]*"|'[^']*'|\([^)]*\))$/.test(rest)) return null;
        return {label: text.slice(index + 1, close), address: address.replace(/\\(.)/g, "$1"), end};
    }

    /** A bare address (GFM's extended autolink) at `index`, its trailing
     *  punctuation left out; null where there is none. */
    function bareUrlAt(text, index) {
        const before = index > 0 ? text[index - 1] : "";
        if (before && !isSpace(before) && "*_~(".indexOf(before) < 0) return null;
        BARE_URL.lastIndex = index;
        const found = BARE_URL.exec(text);
        if (!found) return null;
        let url = found[0];
        for (;;) {
            const last = url[url.length - 1];
            if ("?!.,:*_~'\"".indexOf(last) >= 0) {
                url = url.slice(0, -1);
                continue;
            }
            if (last === ")") {
                const opened = (url.match(/\(/g) || []).length;
                const closed = (url.match(/\)/g) || []).length;
                if (closed > opened) {
                    url = url.slice(0, -1);
                    continue;
                }
            }
            const entity = /&[A-Za-z0-9]+;$/.exec(url);
            if (entity) {
                url = url.slice(0, -entity[0].length);
                continue;
            }
            break;
        }
        const domain = url.replace(/^(?:https?:\/\/|www\.)/i, "").split(/[/?#]/)[0]
            .replace(/:\d+$/, "");
        if (!/^[A-Za-z0-9_-]+(?:\.[A-Za-z0-9_-]+)+$/.test(domain) && !/^localhost$/i.test(domain)) {
            return null;
        }
        return {url, end: index + url.length};
    }

    function flanking(text, start, end) {
        const before = start > 0 ? text[start - 1] : "";
        const after = end < text.length ? text[end] : "";
        const punctuated = (char) => !!char && (PUNCTUATION.test(char) || (char >= "\uE000" && char <= "\uE00F"));
        const spaceBefore = !before || isSpace(before);
        const spaceAfter = !after || isSpace(after);
        const left = !spaceAfter && (!punctuated(after) || spaceBefore || punctuated(before));
        const right = !spaceBefore && (!punctuated(before) || spaceAfter || punctuated(after));
        return {left, right, before: punctuated(before), after: punctuated(after)};
    }

    /** One run of inline text, drawn: CommonMark's emphasis and GFM's `~~`,
     *  code spans, links, autolinks, bare addresses, entities, escapes, the
     *  inline tags of TAG_NAMES and <br>, formulas and quotes. `nested` is a
     *  link's label, where no other link may start. */
    function inline(text, context, nested) {
        const nodes = [];
        let buffer = "";
        const flush = () => {
            if (buffer) nodes.push({t: "text", v: buffer});
            buffer = "";
        };
        const raw = (html) => {
            flush();
            nodes.push({t: "raw", v: html});
        };
        let index = 0;
        while (index < text.length) {
            const char = text[index];
            if (char === MATH_OPEN) {
                const shut = text.indexOf(MATH_CLOSE, index);
                raw(mathHtml(context, Number(text.slice(index + 1, shut))));
                index = shut + 1;
                continue;
            }
            if (char === QUOTE_OPEN || char === QUOTE_CLOSE) {
                flush();
                nodes.push({t: char === QUOTE_OPEN ? "qo" : "qc"});
                index += 1;
                continue;
            }
            if (char === "\\") {
                const next = text[index + 1];
                if (next === "\n") {
                    raw("<br>");
                    index += 2;
                    continue;
                }
                if (next !== undefined && ASCII_PUNCTUATION.indexOf(next) >= 0) {
                    raw(escapeHtml(next));
                    index += 2;
                    continue;
                }
                buffer += char;
                index += 1;
                continue;
            }
            if (char === "`") {
                let run = index;
                while (run < text.length && text[run] === "`") run += 1;
                const stop = backtickRun(text, run, text.length, run - index);
                if (stop < 0) {
                    buffer += text.slice(index, run);
                    index = run;
                    continue;
                }
                let code = text.slice(run, stop - (run - index)).replace(/\n/g, " ");
                if (code.length > 2 && code[0] === " " && code[code.length - 1] === " " && code.trim()) {
                    code = code.slice(1, -1);
                }
                raw("<code>" + escapeHtml(asWritten(code, context)) + "</code>");
                index = stop;
                continue;
            }
            if (char === "*" || char === "_" || char === "~") {
                let run = index;
                while (run < text.length && text[run] === char) run += 1;
                const size = run - index;
                if (char === "~" && size !== 2) {
                    buffer += text.slice(index, run);
                    index = run;
                    continue;
                }
                flush();
                const side = flanking(text, index, run);
                let canOpen = side.left;
                let canClose = side.right;
                if (char === "_") {
                    canOpen = side.left && (!side.right || side.before);
                    canClose = side.right && (!side.left || side.after);
                }
                nodes.push({t: "delim", c: char, n: size, size, canOpen, canClose});
                index = run;
                continue;
            }
            if (char === "!" && text[index + 1] === "[" && !nested) {
                const link = linkAt(text, index + 1);
                if (link) {
                    const url = safeUrl(link.address);
                    const words = escapeHtml(asWritten(link.label, context).trim() || "picture");
                    raw(url ? linkOpen(url) + "\u{1F5BC}\uFE0F " + words + "</a>" : words);
                    index = link.end;
                    continue;
                }
            }
            if (char === "[" && !nested) {
                const link = linkAt(text, index);
                if (link) {
                    const url = safeUrl(link.address);
                    const label = inline(link.label, context, true);
                    raw(url ? linkOpen(url) + label + "</a>" : label);
                    index = link.end;
                    continue;
                }
            }
            if (char === "<") {
                AUTOLINK.lastIndex = index;
                const auto = !nested && AUTOLINK.exec(text);
                if (auto && !/[\uE000-\uE00F]/.test(auto[0])) {
                    const target = auto[0].slice(1, -1);
                    const url = safeUrl(target.indexOf(":") < 0 ? "mailto:" + target : target);
                    raw(url ? linkOpen(url) + escapeHtml(target) + "</a>" : escapeHtml(auto[0]));
                    index += auto[0].length;
                    continue;
                }
                BREAK_TAG.lastIndex = index;
                const broken = BREAK_TAG.exec(text);
                if (broken) {
                    raw("<br>");
                    index += broken[0].length;
                    continue;
                }
                INLINE_TAG.lastIndex = index;
                const tag = INLINE_TAG.exec(text);
                if (tag && TAG_NAMES.has(tag[2].toLowerCase())) {
                    flush();
                    nodes.push({t: tag[1] ? "tc" : "to", name: tag[2].toLowerCase(), v: tag[0]});
                    index += tag[0].length;
                    continue;
                }
            }
            if (char === "&") {
                const entity = entityAt(text, index);
                if (entity) {
                    raw(escapeHtml(entity.char));
                    index = entity.end;
                    continue;
                }
            }
            if ((char === "h" || char === "H" || char === "w" || char === "W") && !nested) {
                const bare = bareUrlAt(text, index);
                if (bare) {
                    const url = safeUrl(/^www\./i.test(bare.url) ? "http://" + bare.url : bare.url);
                    if (url) {
                        raw(linkOpen(url) + escapeHtml(bare.url) + "</a>");
                        index = bare.end;
                        continue;
                    }
                }
            }
            buffer += char;
            index += 1;
        }
        flush();
        emphasis(nodes);
        return draw(pairUp(nodes));
    }

    /** CommonMark's "process emphasis" (6.2), over a flat list of nodes:
     *  each closing run takes the nearest opening run of its character it may
     *  pair with, two characters for <strong> where both have two, the rule of
     *  three kept; GFM's `~~` the same way, for <del>. What is between them
     *  becomes the new element's children. */
    function emphasis(nodes) {
        for (let closer = 0; closer < nodes.length; closer += 1) {
            const close = nodes[closer];
            if (close.t !== "delim" || !close.canClose || close.n === 0) continue;
            let opener = -1;
            for (let at = closer - 1; at >= 0; at -= 1) {
                const open = nodes[at];
                if (open.t !== "delim" || open.c !== close.c || !open.canOpen || open.n === 0) continue;
                const odd = (open.canClose || close.canOpen) && (open.size + close.size) % 3 === 0
                    && !(open.size % 3 === 0 && close.size % 3 === 0);
                if (close.c !== "~" && odd) continue;
                opener = at;
                break;
            }
            if (opener < 0) continue;
            const open = nodes[opener];
            const use = close.c === "~" ? 2 : (open.n >= 2 && close.n >= 2 ? 2 : 1);
            const tag = close.c === "~" ? "del" : (use === 2 ? "strong" : "em");
            open.n -= use;
            close.n -= use;
            const inside = nodes.splice(opener + 1, closer - opener - 1);
            inside.forEach((node) => {
                if (node.t === "delim") node.canOpen = node.canClose = false;
            });
            nodes.splice(opener + 1, 0, {t: "wrap", tag, children: inside});
            // The closing run is now at opener + 2, where the loop looks
            // next: with characters left, it may close again.
            closer = opener + 1;
        }
    }

    /** Quotes and inline tags, paired among siblings, so the HTML is always
     *  properly nested: a pair whose two ends are not in the same element is
     *  drawn as text (a tag) or not marked (a quote), and where a tag and a
     *  quote cross, the tag -- the formatting the model wrote -- is kept and
     *  the quote is not marked. */
    function pairUp(nodes) {
        const out = [];
        const stack = [];
        nodes.forEach((node) => {
            if (node.t === "wrap") node.children = pairUp(node.children);
            if (node.t === "qo" || node.t === "to") {
                stack.push({node, at: out.length});
                out.push(node);
                return;
            }
            if (node.t === "qc" || node.t === "tc") {
                let found = -1;
                for (let at = stack.length - 1; at >= 0; at -= 1) {
                    const open = stack[at].node;
                    if (node.t === "qc" ? open.t === "qo" : (open.t === "to" && open.name === node.name)) {
                        found = at;
                        break;
                    }
                }
                if (found < 0) {
                    out.push(node.t === "tc" ? {t: "text", v: node.v} : {t: "skip"});
                    return;
                }
                if (node.t === "qc" && found < stack.length - 1) {
                    // A tag opened inside the quote is still open: the quote
                    // gives way, and the tag may still find its end.
                    stack.splice(found, 1);
                    out.push({t: "skip"});
                    return;
                }
                const start = stack[found].at;
                const open = stack[found].node;
                stack.length = found;
                const children = out.splice(start + 1);
                out.splice(start, 1, {t: "wrap", tag: open.t === "qo" ? "q" : open.name, children});
                return;
            }
            out.push(node);
        });
        // Openings never closed: a tag is its text, a quote is not marked.
        return out.map((node) => (node.t === "to" ? {t: "text", v: node.v}
                                  : (node.t === "qo" ? {t: "skip"} : node)));
    }

    function draw(nodes) {
        return nodes.map((node) => {
            if (node.t === "text") return escapeHtml(node.v).replace(/[ \t]+\n/g, "\n").replace(/\n/g, "<br>");
            if (node.t === "raw") return node.v;
            // A tag left open inside an element that has closed: its text.
            if (node.t === "to" || node.t === "tc") return escapeHtml(node.v);
            if (node.t === "delim") return escapeHtml(node.c.repeat(node.n));
            if (node.t === "wrap") {
                const open = node.tag === "q" ? '<q class="forge-assistant-quote">' : "<" + node.tag + ">";
                return open + draw(node.children) + "</" + node.tag + ">";
            }
            return "";
        }).join("");
    }

    /** A message's words as HTML: the reading above, then blocks and inlines. */
    function renderMarkdown(text) {
        const source = normalised(text).replace(OURS, "\ufffd");
        const context = {maths: [], wantsKatex: false};
        const parts = chunks(source).map((run) => {
            if (run[0]) return run[1];
            const body = run[1];
            const found = atoms(body);
            const quoted = marks(body, found)[0];
            const opening = new Set(quoted.map((pair) => pair[0]));
            const closing = new Set(quoted.map((pair) => pair[1]));
            const mathAt = new Map(found.filter((atom) => atom[2] === MATH).map((atom) => [atom[0], atom]));
            let out = "";
            let index = 0;
            while (index < body.length) {
                const atom = mathAt.get(index);
                if (atom) {
                    context.maths.push({tex: atom[3][0], display: atom[3][1],
                                        source: body.slice(atom[0], atom[1])});
                    out += MATH_OPEN + (context.maths.length - 1) + MATH_CLOSE;
                    index = atom[1];
                    continue;
                }
                if (opening.has(index)) out += QUOTE_OPEN;
                out += body[index];
                if (closing.has(index)) out += QUOTE_CLOSE;
                index += 1;
            }
            return out;
        });
        const html = renderBlocks(parts.join("\n").split("\n"), context, false);
        if (context.wantsKatex) loadKatex();
        return html;
    }

    NS.renderMarkdown = renderMarkdown;
    NS.safeUrl = safeUrl;
    NS.describeMessage = describeMessage;
    NS.typesetMath = typesetPending;
    NS.useKatex = useKatex;
    NS.loadKatex = loadKatex;
    NS.KATEX_VERSION = KATEX_VERSION;
})();
