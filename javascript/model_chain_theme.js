// Model Chain -- the theme's page work.
//
// The palette is a Gradio theme object, installed by mc_theme.py when the
// page is built; this file does what a palette cannot: it puts Forge's tab
// bar in a 64 px bar at the top with a title, a theme toggle, a settings
// button and a menu for the tabs that do not fit; moves Generate above the
// gallery and the prompt boxes to the top of the settings column; moves the
// quick settings into a drawer; marks the prompt boxes resizable; and draws
// icons on the tool buttons in place of their emoji. Each of those is a
// switch under Settings -> Model Chain: Theme, read live, and each is undone
// when its switch goes off: every move remembers where its element came
// from, every icon remembers its text, and nothing is written on an element
// that a reload would not also write.
//
// Nothing here comes from sd-webui-lobe-theme. The page it produces is the
// one the user had under that theme, described (handoff 43, docs/24) and
// rebuilt from Forge's own DOM: Gradio's `#tabs > .tab-nav` with its tab
// buttons, Forge's `txt2img_toprow`, `txt2img_generate_box`,
// `txt2img_gallery_container` and `#quicksettings`, and the emoji Forge puts
// on its tool buttons. Forge's own script reads the current tab from
// `#tabs > .tab-nav > button.selected`, a direct-child selector, so the bar
// is never moved or wrapped: the title and the actions go *into* it, before
// and after Gradio's buttons.
//
// Standing down. While the Lobe extension's own bundle is on the page -- a
// `#root` under `gradio-app` holding a tablist that is not Forge's -- the
// page is that theme's, and this does nothing but say so on the root
// (`data-mc-theme="lobe"`); the user turns this theme on first and disables
// that one second, and nothing fights in between.
//
// The icons are Lucide's (https://lucide.dev), under the ISC licence:
//
//   Copyright (c) 2026 Lucide Icons and Contributors
//   Permission to use, copy, modify, and/or distribute this software for any
//   purpose with or without fee is hereby granted, provided that the above
//   copyright notice and this permission notice appear in all copies.
//   THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
//   WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
//   MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
//   ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
//   WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
//   ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
//   OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
//
// Of these, arrow-down-left and trash are derived from Feather (MIT,
// Copyright (c) 2013-present Cole Bemis).
(function () {
    "use strict";

    // -- the settings, by the names Python registers ------------------------ //

    const OPTION = {
        enabled: "model_chain_theme",
        split: "model_chain_theme_split",
        scroll: "model_chain_theme_scroll",
        quick: "model_chain_theme_quick",
        quickWidth: "model_chain_theme_quick_width",
        prompts: "model_chain_theme_prompts",
        icons: "model_chain_theme_icons",
        title: "model_chain_theme_title",
        labels: "model_chain_theme_tab_labels",
        confirm: "model_chain_theme_confirm_unload",
        font: "model_chain_theme_font",
        networks: "model_chain_theme_networks",
        networksWidth: "model_chain_theme_networks_width",
        networksCard: "model_chain_theme_networks_card",
        networksMode: "model_chain_theme_networks_mode",
        info: "model_chain_theme_image_info",
    };
    const DEFAULTS = {
        enabled: false, split: true, scroll: true, quick: "collapsed", quickWidth: 280, prompts: true,
        icons: true, title: "Forge", labels: "as_forge", confirm: false, font: "",
        networks: "collapsed", networksWidth: 340, networksCard: 120, networksMode: "drawer",
        info: true,
    };
    const NETWORKS_WIDTH_MIN = 240;
    const NETWORKS_WIDTH_MAX = 800;
    const NETWORKS_CARD_MIN = 60;
    const NETWORKS_CARD_MAX = 320;
    const NETWORKS_LAYOUTS = ["drawer", "column"];
    const QUICK_MODES = ["off", "collapsed", "open"];
    const LABEL_MODES = ["as_forge", "spaced"];
    const WIDTH_MIN = 200;
    const WIDTH_MAX = 600;

    // -- what is written on the page ---------------------------------------- //

    const ATTRIBUTE = "data-mc-theme";            // "on", or "lobe" while standing down
    const PROMPTS_ATTRIBUTE = "data-mc-theme-prompts";
    const CACHE = "mc-theme";                     // the last config, for the first paint
    const QUICK_STATE = "mc-theme-quick";         // "open" | "closed", per browser
    const NETWORKS_STATE = "mc-theme-networks";   // the same, for the right-hand drawer
    const NETWORKS_ID = "mc-theme-networks";          // the id prefix of a tab's drawer head
    const NETWORKS_NAV_CLASS = "mc-theme-networks-nav";   // Forge's extra-network tab strip, out of the way
    const NETWORKS_SHOWN_CLASS = "mc-theme-networks-shown"; // the one page the drawer shows
    const INFO_CLASS = "mc-theme-info";
    const INFO_SOURCE_CLASS = "mc-theme-info-source";       // Forge's own infotext, drawn as the panel instead
    const BAR_CLASS = "mc-theme-bar";
    const OVERFLOW_CLASS = "mc-theme-overflow";   // a tab button the bar has no room for
    const SETTINGS_SECTION = "Model Chain: Theme";
    const QUICK_ID = "mc-theme-quick";
    const FONT_LINK_ID = "mc-theme-font";
    const HEADER_GAP_PX = 4;

    // -- the icons ------------------------------------------------------------ //

    const SVG = 'xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" '
              + 'stroke="currentColor" stroke-width="2" stroke-linecap="round" '
              + 'stroke-linejoin="round" aria-hidden="true" focusable="false"';
    function icon(paths) { return "<svg " + SVG + ">" + paths + "</svg>"; }

    const GLYPH = {
        "arrow-down-left": icon('<path d="M17 7 7 17"/><path d="M17 17H7V7"/>'),
        "trash": icon('<path d="M10 11v6"/><path d="M14 11v6"/><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6"/><path d="M3 6h18"/><path d="M8 6V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/>'),
        "clipboard-list": icon('<rect width="8" height="4" x="8" y="2" rx="1" ry="1"/><path d="M16 4h2a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h2"/><path d="M12 11h4"/><path d="M12 16h4"/><path d="M8 11h.01"/><path d="M8 16h.01"/>'),
        "refresh-cw": icon('<path d="M3 12a9 9 0 0 1 9-9 9.75 9.75 0 0 1 6.74 2.74L21 8"/><path d="M21 3v5h-5"/><path d="M21 12a9 9 0 0 1-9 9 9.75 9.75 0 0 1-6.74-2.74L3 16"/><path d="M8 16H3v5"/>'),
        "rotate-ccw": icon('<path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8"/><path d="M3 3v5h5"/>'),
        "dices": icon('<rect width="12" height="12" x="2" y="10" rx="2" ry="2"/><path d="m17.92 14 3.5-3.5a2.24 2.24 0 0 0 0-3l-5-4.92a2.24 2.24 0 0 0-3 0L10 6"/><path d="M6 18h.01"/><path d="M10 14h.01"/><path d="M15 6h.01"/><path d="M18 9h.01"/>'),
        "recycle": icon('<path d="M7 19H4.815a1.83 1.83 0 0 1-1.57-.881 1.785 1.785 0 0 1-.004-1.784L7.196 9.5"/><path d="M11 19h8.203a1.83 1.83 0 0 0 1.556-.89 1.784 1.784 0 0 0 0-1.775l-1.226-2.12"/><path d="m14 16-3 3 3 3"/><path d="M8.293 13.596 7.196 9.5 3.1 10.598"/><path d="m9.344 5.811 1.093-1.892A1.83 1.83 0 0 1 11.985 3a1.784 1.784 0 0 1 1.546.888l3.943 6.843"/><path d="m13.378 9.633 4.096 1.098 1.097-4.096"/>'),
        "folder": icon('<path d="M20 20a2 2 0 0 0 2-2V8a2 2 0 0 0-2-2h-7.9a2 2 0 0 1-1.69-.9L9.6 3.9A2 2 0 0 0 7.93 3H4a2 2 0 0 0-2 2v13a2 2 0 0 0 2 2Z"/>'),
        "save": icon('<path d="M15.2 3a2 2 0 0 1 1.4.6l3.8 3.8a2 2 0 0 1 .6 1.4V19a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2z"/><path d="M17 21v-7a1 1 0 0 0-1-1H8a1 1 0 0 0-1 1v7"/><path d="M7 3v4a1 1 0 0 0 1 1h7"/>'),
        "file-archive": icon('<path d="M13.659 22H18a2 2 0 0 0 2-2V8a2.4 2.4 0 0 0-.706-1.706l-3.588-3.588A2.4 2.4 0 0 0 14 2H6a2 2 0 0 0-2 2v11.5"/><path d="M14 2v5a1 1 0 0 0 1 1h5"/><path d="M8 12v-1"/><path d="M8 18v-2"/><path d="M8 7V6"/><circle cx="8" cy="20" r="2"/>'),
        "image": icon('<rect width="18" height="18" x="3" y="3" rx="2" ry="2"/><circle cx="9" cy="9" r="2"/><path d="m21 15-3.086-3.086a2 2 0 0 0-2.828 0L6 21"/>'),
        "paintbrush": icon('<path d="m14.622 17.897-10.68-2.913"/><path d="M18.376 2.622a1 1 0 1 1 3.002 3.002L17.36 9.643a.5.5 0 0 0 0 .707l.944.944a2.41 2.41 0 0 1 0 3.408l-.944.944a.5.5 0 0 1-.707 0L8.354 7.348a.5.5 0 0 1 0-.707l.944-.944a2.41 2.41 0 0 1 3.408 0l.944.944a.5.5 0 0 0 .707 0z"/><path d="M9 8c-1.804 2.71-3.97 3.46-6.583 3.948a.507.507 0 0 0-.302.819l7.32 8.883a1 1 0 0 0 1.185.204C12.735 20.405 16 16.792 16 15"/>'),
        "ruler": icon('<path d="M21.3 15.3a2.4 2.4 0 0 1 0 3.4l-2.6 2.6a2.4 2.4 0 0 1-3.4 0L2.7 8.7a2.41 2.41 0 0 1 0-3.4l2.6-2.6a2.41 2.41 0 0 1 3.4 0Z"/><path d="m14.5 12.5 2-2"/><path d="m11.5 9.5 2-2"/><path d="m8.5 6.5 2-2"/><path d="m17.5 15.5 2-2"/>'),
        "sparkles": icon('<path d="M11.017 2.814a1 1 0 0 1 1.966 0l1.051 5.558a2 2 0 0 0 1.594 1.594l5.558 1.051a1 1 0 0 1 0 1.966l-5.558 1.051a2 2 0 0 0-1.594 1.594l-1.051 5.558a1 1 0 0 1-1.966 0l-1.051-5.558a2 2 0 0 0-1.594-1.594l-5.558-1.051a1 1 0 0 1 0-1.966l5.558-1.051a2 2 0 0 0 1.594-1.594z"/><path d="M20 2v4"/><path d="M22 4h-4"/><circle cx="4" cy="20" r="2"/>'),
        "arrow-up-down": icon('<path d="m21 16-4 4-4-4"/><path d="M17 20V4"/><path d="m3 8 4-4 4 4"/><path d="M7 4v16"/>'),
        "wand-sparkles": icon('<path d="m21.64 3.64-1.28-1.28a1.21 1.21 0 0 0-1.72 0L2.36 18.64a1.21 1.21 0 0 0 0 1.72l1.28 1.28a1.2 1.2 0 0 0 1.72 0L21.64 5.36a1.2 1.2 0 0 0 0-1.72"/><path d="m14 7 3 3"/><path d="M5 6v4"/><path d="M19 14v4"/><path d="M10 2v2"/><path d="M7 8H3"/><path d="M21 16h-4"/><path d="M11 3H9"/>'),
        "pencil-line": icon('<path d="M13 21h8"/><path d="m15 5 4 4"/><path d="M21.174 6.812a1 1 0 0 0-3.986-3.987L3.842 16.174a2 2 0 0 0-.5.83l-1.321 4.352a.5.5 0 0 0 .623.622l4.353-1.32a2 2 0 0 0 .83-.497z"/>'),
        "notebook-pen": icon('<path d="M13.4 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-7.4"/><path d="M2 6h4"/><path d="M2 10h4"/><path d="M2 14h4"/><path d="M2 18h4"/><path d="M21.378 5.626a1 1 0 1 0-3.004-3.004l-5.01 5.012a2 2 0 0 0-.506.854l-.837 2.87a.5.5 0 0 0 .62.62l2.87-.837a2 2 0 0 0 .854-.506z"/>'),
        "layers": icon('<path d="M12.83 2.18a2 2 0 0 0-1.66 0L2.6 6.08a1 1 0 0 0 0 1.83l8.58 3.91a2 2 0 0 0 1.66 0l8.58-3.9a1 1 0 0 0 0-1.83z"/><path d="M2 12a1 1 0 0 0 .58.91l8.6 3.91a2 2 0 0 0 1.65 0l8.58-3.9A1 1 0 0 0 22 12"/><path d="M2 17a1 1 0 0 0 .58.91l8.6 3.91a2 2 0 0 0 1.65 0l8.58-3.9A1 1 0 0 0 22 17"/>'),
        "x": icon('<path d="M18 6 6 18"/><path d="m6 6 12 12"/>'),
        "sun": icon('<circle cx="12" cy="12" r="4"/><path d="M12 2v2"/><path d="M12 20v2"/><path d="m4.93 4.93 1.41 1.41"/><path d="m17.66 17.66 1.41 1.41"/><path d="M2 12h2"/><path d="M20 12h2"/><path d="m6.34 17.66-1.41 1.41"/><path d="m19.07 4.93-1.41 1.41"/>'),
        "moon": icon('<path d="M20.985 12.486a9 9 0 1 1-9.473-9.472c.405-.022.617.46.402.803a6 6 0 0 0 8.268 8.268c.344-.215.825-.004.803.401"/>'),
        "settings": icon('<path d="M9.671 4.136a2.34 2.34 0 0 1 4.659 0 2.34 2.34 0 0 0 3.319 1.915 2.34 2.34 0 0 1 2.33 4.033 2.34 2.34 0 0 0 0 3.831 2.34 2.34 0 0 1-2.33 4.033 2.34 2.34 0 0 0-3.319 1.915 2.34 2.34 0 0 1-4.659 0 2.34 2.34 0 0 0-3.32-1.915 2.34 2.34 0 0 1-2.33-4.033 2.34 2.34 0 0 0 0-3.831A2.34 2.34 0 0 1 6.35 6.051a2.34 2.34 0 0 0 3.319-1.915"/><circle cx="12" cy="12" r="3"/>'),
        "sliders-horizontal": icon('<path d="M10 5H3"/><path d="M12 19H3"/><path d="M14 3v4"/><path d="M16 17v4"/><path d="M21 12h-9"/><path d="M21 19h-5"/><path d="M21 5h-7"/><path d="M8 10v4"/><path d="M8 12H3"/>'),
        "ellipsis": icon('<circle cx="12" cy="12" r="1"/><circle cx="19" cy="12" r="1"/><circle cx="5" cy="12" r="1"/>'),
    };

    // Forge's tool-button emoji (modules/ui.py, ui_prompt_styles.py,
    // ui_common.py), each to a glyph. Variation selectors are stripped on both
    // sides, so "🗑️" and "🗑" are the same key.
    const ICONS = {
        "↙": "arrow-down-left",    // paste
        "🗑": "trash",              // clear prompt
        "📋": "clipboard-list",     // apply styles
        "🔄": "refresh-cw",         // refresh
        "🌀": "rotate-ccw",         // restore progress
        "🎲": "dices",              // random seed
        "♻": "recycle",            // reuse seed
        "📂": "folder",             // open folder
        "💾": "save",               // save
        "🗃": "file-archive",       // save zip
        "🖼": "image",              // send to img2img
        "🎨": "paintbrush",         // send to inpaint
        "📐": "ruler",              // send to extras / detect size
        "✨": "sparkles",           // upscale
        "⇅": "arrow-up-down",      // switch width and height
        "🖌": "pencil-line",        // edit styles
        "📝": "notebook-pen",       // copy prompt to style
        "🎴": "layers",             // extra networks
        "🪄": "wand-sparkles",      // a wand, wherever one is drawn
    };

    function plain(text) {
        return String(text || "").replace(/[︎️]/g, "").trim();
    }

    function iconFor(text) {
        const name = ICONS[plain(text)];
        return name ? GLYPH[name] : null;
    }

    // -- reading the settings ------------------------------------------------- //

    function setting(name, fallback) {
        try {
            if (typeof opts === "undefined" || opts === null) return fallback;
            const value = opts[name];
            return value === undefined || value === null ? fallback : value;
        } catch (error) {
            return fallback;
        }
    }

    function normalise(raw) {
        raw = raw || {};
        const out = Object.assign({}, DEFAULTS);
        out.enabled = !!raw.enabled;
        out.split = raw.split === undefined ? DEFAULTS.split : !!raw.split;
        out.scroll = raw.scroll === undefined ? DEFAULTS.scroll : !!raw.scroll;
        out.quick = QUICK_MODES.indexOf(raw.quick) === -1 ? DEFAULTS.quick : raw.quick;
        const width = Number(raw.quickWidth);
        out.quickWidth = Number.isFinite(width)
            ? Math.min(WIDTH_MAX, Math.max(WIDTH_MIN, Math.round(width))) : DEFAULTS.quickWidth;
        out.prompts = raw.prompts === undefined ? DEFAULTS.prompts : !!raw.prompts;
        out.icons = raw.icons === undefined ? DEFAULTS.icons : !!raw.icons;
        const title = raw.title === undefined || raw.title === null ? DEFAULTS.title : String(raw.title);
        out.title = title.trim();
        out.labels = LABEL_MODES.indexOf(raw.labels) === -1 ? DEFAULTS.labels : raw.labels;
        out.confirm = !!raw.confirm;
        out.font = raw.font === undefined || raw.font === null ? "" : String(raw.font).trim();
        out.networks = QUICK_MODES.indexOf(raw.networks) === -1 ? DEFAULTS.networks : raw.networks;
        const networksWidth = Number(raw.networksWidth);
        out.networksWidth = Number.isFinite(networksWidth)
            ? Math.min(NETWORKS_WIDTH_MAX, Math.max(NETWORKS_WIDTH_MIN, Math.round(networksWidth)))
            : DEFAULTS.networksWidth;
        const card = Number(raw.networksCard);
        out.networksCard = Number.isFinite(card)
            ? Math.min(NETWORKS_CARD_MAX, Math.max(NETWORKS_CARD_MIN, Math.round(card)))
            : DEFAULTS.networksCard;
        out.networksMode = NETWORKS_LAYOUTS.indexOf(raw.networksMode) === -1
            ? DEFAULTS.networksMode : raw.networksMode;
        out.info = raw.info === undefined ? DEFAULTS.info : !!raw.info;
        return out;
    }

    function read() {
        const raw = {};
        Object.keys(OPTION).forEach((key) => { raw[key] = setting(OPTION[key], DEFAULTS[key]); });
        return normalise(raw);
    }

    function remember(config) {
        try { localStorage.setItem(CACHE, JSON.stringify(config)); } catch (error) { /* a private window */ }
    }

    function recall() {
        try {
            const text = localStorage.getItem(CACHE);
            return text ? normalise(JSON.parse(text)) : null;
        } catch (error) {
            return null;
        }
    }

    // -- the page ------------------------------------------------------------- //

    function app() {
        try {
            if (typeof gradioApp === "function") return gradioApp() || document;
        } catch (error) { /* fall through */ }
        return document;
    }

    function root() { return document.documentElement; }

    /** Puts a node of this theme's own -- a drawer -- inside Forge's app
     *  element. Forge's scripts, and this one, find things through
     *  gradioApp().querySelector, which never looks outside that element: a
     *  drawer on the body took the quick settings out of Forge's reach and
     *  out of this script's, which then put them back, took them again, and
     *  so on at every update of the page. */
    function mount(node) {
        const host = app();
        const inside = host && host !== document && typeof host.appendChild === "function";
        (inside ? host : document.body).appendChild(node);
    }

    function query(selector, scope) {
        try { return (scope || app()).querySelector(selector); } catch (error) { return null; }
    }

    function queryAll(selector, scope) {
        try { return Array.prototype.slice.call((scope || app()).querySelectorAll(selector)); }
        catch (error) { return []; }
    }

    /** Lobe's bundle on the page: a #root under gradio-app with a tablist of
     *  its own, outside Forge's #tabs. */
    function lobeRoot() {
        const found = query("#root");
        if (!found) return null;
        const strip = query("[role='tablist']", found);
        if (!strip) return null;
        if (strip.closest && strip.closest("#tabs")) return null;
        return found;
    }

    function depth(node, stop) {
        let count = 0;
        let walk = node;
        while (walk && walk !== stop) { count += 1; walk = walk.parentElement; }
        return count;
    }

    /** Forge's top-level tab bar: the shallowest .tab-nav under #tabs. */
    function bar() {
        const tabs = query("#tabs");
        if (!tabs) return null;
        const bars = queryAll(".tab-nav, [role='tablist']", tabs);
        if (!bars.length) return null;
        let strip = bars[0];
        let shallowest = depth(strip, tabs);
        bars.forEach((candidate) => {
            const found = depth(candidate, tabs);
            if (found < shallowest) { strip = candidate; shallowest = found; }
        });
        return strip;
    }

    function tabButtons(strip) {
        return Array.prototype.slice.call(strip.children).filter(
            (child) => child.tagName === "BUTTON" && child.getAttribute("role") === "tab");
    }

    function attempt(what, run) {
        try { run(); } catch (error) { console.error("Model Chain theme: could not " + what, error); }
    }

    function make(tag, className, attrs) {
        const node = document.createElement(tag);
        if (className) node.className = className;
        Object.keys(attrs || {}).forEach((name) => node.setAttribute(name, attrs[name]));
        return node;
    }

    // -- state ----------------------------------------------------------------- //

    const state = {
        moved: [],          // {node, parent, next, key}: where a moved element came from
        swapped: [],        // buttons whose text became an icon
        relabelled: [],     // tab buttons whose text was respaced
        header: null,       // {strip, title, actions, more, menu, quickToggle}
        quick: null,        // the drawer
        networks: null,     // {hosts: {tabname: {host, nav, chrome, strip, controls, pages, page, observer}}, open, width, setting, mode}
        infoPanels: new Map(), // Forge's infotext element -> {panel, html}
        scroll: null,       // {entries: {tabname: {row, column, results}}, observer, observed, body, frame}
        unload: null,       // the beforeunload handler while asked for
        wired: false,
        fitPending: false,
        lastLobe: false,
    };

    // -- the header ------------------------------------------------------------- //

    function isDark() {
        return !!(document.body && document.body.classList.contains("dark"))
            || !!query(".dark", document);
    }

    /** The page's address with the other theme: Forge's set_theme() adds the
     *  query only when there is none, so the toggle rewrites it itself. */
    function nextThemeUrl(href, dark) {
        const url = new URL(href);
        url.searchParams.set("__theme", dark ? "light" : "dark");
        return url.toString();
    }

    function action(className, glyph, label) {
        const button = make("button", "mc-theme-action " + className,
                            {type: "button", title: label, "aria-label": label});
        button.innerHTML = GLYPH[glyph];
        return button;
    }

    function openSettings() {
        const strip = bar();
        if (!strip) return;
        const settingsTab = tabButtons(strip).find(
            (button) => button.getAttribute("aria-controls") === "tab_settings");
        if (settingsTab) settingsTab.click();
        setTimeout(function () {
            const inner = queryAll("#settings .tab-nav button, #tab_settings .tab-nav button")
                .find((button) => plain(button.textContent) === SETTINGS_SECTION);
            if (inner) {
                inner.click();
                if (inner.scrollIntoView) inner.scrollIntoView({block: "nearest"});
            }
        }, 60);
    }

    function closeMenu() {
        if (state.header && state.header.menu) {
            state.header.menu.hidden = true;
            state.header.more.setAttribute("aria-expanded", "false");
        }
    }

    function toggleMenu() {
        const header = state.header;
        if (!header) return;
        const open = header.menu.hidden;
        header.menu.hidden = !open;
        header.more.setAttribute("aria-expanded", open ? "true" : "false");
    }

    function installHeader(strip, config) {
        setClass(strip, BAR_CLASS, true);
        let header = state.header;
        if (!header || header.strip !== strip) {
            if (header) removeHeader();
            const title = make("div", "mc-theme-title");
            const actions = make("div", "mc-theme-actions");
            const quickToggle = action("mc-theme-quick-toggle", "sliders-horizontal", "Quick settings");
            quickToggle.setAttribute("aria-expanded", "false");
            quickToggle.setAttribute("aria-controls", QUICK_ID);
            const networksToggle = action("mc-theme-networks-toggle", "layers", "Extra networks");
            networksToggle.setAttribute("aria-expanded", "false");
            const more = action("mc-theme-more", "ellipsis", "More tabs");
            more.setAttribute("aria-haspopup", "menu");
            more.setAttribute("aria-expanded", "false");
            const mode = action("mc-theme-mode", isDark() ? "sun" : "moon",
                                isDark() ? "Switch to the light theme" : "Switch to the dark theme");
            const settings = action("mc-theme-settings", "settings", "Theme settings");
            const menu = make("div", "mc-theme-menu", {role: "menu"});
            menu.hidden = true;
            actions.appendChild(quickToggle);
            actions.appendChild(networksToggle);
            actions.appendChild(more);
            actions.appendChild(mode);
            actions.appendChild(settings);
            actions.appendChild(menu);
            strip.insertBefore(title, strip.firstChild);
            strip.appendChild(actions);
            quickToggle.addEventListener("click", function () { toggleQuick(); });
            networksToggle.addEventListener("click", function () { toggleNetworks(); });
            more.addEventListener("click", function (event) { event.stopPropagation(); toggleMenu(); });
            mode.addEventListener("click", function () {
                try { location.replace(nextThemeUrl(location.href, isDark())); } catch (error) { /* a stub */ }
            });
            settings.addEventListener("click", openSettings);
            header = state.header = {strip, title, actions, more, menu, quickToggle, networksToggle};
        }
        // Written only when they differ. Forge runs every after-update
        // callback 250 ms after any change to a child list inside its app
        // element, and this is one of those callbacks: writing the same
        // title is a new text node to that observer, and the first build
        // wrote it on every apply -- the page updated every 300 ms, on every
        // tab, for as long as it was open (`fit` had the same fault).
        if (header.title.textContent !== config.title) header.title.textContent = config.title;
        show(header.title, !!config.title);
        show(header.quickToggle, config.quick !== "off");
        show(header.networksToggle, config.networks !== "off");
    }

    /** `hidden`, written only when it changes: an attribute written to the
     *  same value still wakes every observer of it. */
    function show(node, shown) {
        if (node.hidden !== !shown) node.hidden = !shown;
    }

    /** A class put on or taken off only when that changes, for the same
     *  reason. */
    function setClass(node, name, on) {
        if (node.classList.contains(name) === !!on) return;
        if (on) node.classList.add(name); else node.classList.remove(name);
    }

    /** An attribute written only when it changes, for the same reason. */
    function setAttr(node, name, value) {
        if (node.getAttribute(name) !== value) node.setAttribute(name, value);
    }

    function removeHeader() {
        const header = state.header;
        if (!header) return;
        header.strip.classList.remove(BAR_CLASS);
        tabButtons(header.strip).forEach((button) => button.classList.remove(OVERFLOW_CLASS));
        if (header.title.parentElement) header.title.parentElement.removeChild(header.title);
        if (header.actions.parentElement) header.actions.parentElement.removeChild(header.actions);
        state.header = null;
    }

    /** Which tab buttons fit: the first ones, in order, as far as the room
     *  goes -- and the selected one always, at the expense of the last one
     *  that fitted. Pure, so the node tests hold it. */
    function plan(widths, available, selected) {
        const shown = [];
        let used = 0;
        for (let i = 0; i < widths.length; i += 1) {
            if (used + widths[i] <= available) { shown.push(i); used += widths[i]; }
            else break;
        }
        if (selected >= 0 && selected < widths.length && shown.indexOf(selected) === -1) {
            while (shown.length && used + widths[selected] > available) {
                used -= widths[shown.pop()];
            }
            if (used + widths[selected] <= available) { shown.push(selected); used += widths[selected]; }
        }
        const hidden = [];
        for (let i = 0; i < widths.length; i += 1) if (shown.indexOf(i) === -1) hidden.push(i);
        return {shown, hidden};
    }

    function fit() {
        const header = state.header;
        if (!header || !header.strip.isConnected) return;
        const strip = header.strip;
        const buttons = tabButtons(strip);
        if (!buttons.length) return;
        // Every button shown for the measurement: a hidden one has no width.
        buttons.forEach((button) => setClass(button, OVERFLOW_CLASS, false));
        show(header.more, true);
        const widths = buttons.map((button) => button.getBoundingClientRect().width + HEADER_GAP_PX);
        const style = window.getComputedStyle ? window.getComputedStyle(strip) : null;
        const padding = style ? (parseFloat(style.paddingLeft) || 0) + (parseFloat(style.paddingRight) || 0) : 0;
        const fixed = header.title.getBoundingClientRect().width + header.actions.getBoundingClientRect().width
                    + 2 * HEADER_GAP_PX;
        const available = strip.getBoundingClientRect().width - padding - fixed;
        const selected = buttons.findIndex((button) => button.classList.contains("selected"));
        const result = plan(widths, available, selected);
        result.hidden.forEach((i) => setClass(buttons[i], OVERFLOW_CLASS, true));
        show(header.more, result.hidden.length > 0);
        if (result.hidden.length === 0) show(header.menu, false);
        // The menu is rebuilt only when what it would hold changed: the same
        // buttons under the same names. Emptying and refilling it on every
        // fit was a child-list change Forge answered with the next
        // after-update, which fitted again, without end (see installHeader).
        const hidden = result.hidden.map((i) => buttons[i]);
        const was = header.menuFor || [];
        const same = was.length === hidden.length
            && was.every((entry, i) => entry.button === hidden[i] && entry.label === hidden[i].textContent);
        if (same) return;
        header.menuFor = hidden.map((button) => ({button, label: button.textContent}));
        while (header.menu.firstChild) header.menu.removeChild(header.menu.firstChild);
        hidden.forEach((button) => {
            const item = make("button", "mc-theme-menu-item", {type: "button", role: "menuitem"});
            item.textContent = button.textContent;
            item.addEventListener("click", function (event) {
                event.stopPropagation();
                closeMenu();
                button.click();
            });
            header.menu.appendChild(item);
        });
    }

    function scheduleFit() {
        if (state.fitPending) return;
        state.fitPending = true;
        const run = function () { state.fitPending = false; attempt("fit the tab bar", fit); };
        if (typeof requestAnimationFrame === "function") requestAnimationFrame(run); else run();
    }

    // -- the tab names ---------------------------------------------------------- //

    /** "txt2img" -> "Txt 2 Img", "LLM Studio" -> "LLM Studio", "LoRA" -> "LoRA":
     *  a space before a capital that follows a lowercase letter and before and
     *  after a run of digits, and a capital at the start of every word. A run
     *  of capitals stays a run. */
    function spaced(label) {
        const text = String(label || "").trim();
        if (!text) return text;
        const parts = text
            .replace(/([a-z])([A-Z][a-z])/g, "$1 $2")
            .replace(/([A-Za-z])(\d)/g, "$1 $2")
            .replace(/(\d)([A-Za-z])/g, "$1 $2")
            .split(/\s+/);
        return parts.map((word) => word.charAt(0).toUpperCase() + word.slice(1)).join(" ");
    }

    function installLabels(strip, config) {
        const buttons = tabButtons(strip);
        if (config.labels !== "spaced") { restoreLabels(); return; }
        buttons.forEach((button) => {
            if (button.dataset.mcThemeLabel !== undefined) return;
            const original = button.textContent;
            const wanted = spaced(original);
            if (wanted === original) return;
            button.dataset.mcThemeLabel = original;
            button.textContent = wanted;
            state.relabelled.push(button);
        });
    }

    function restoreLabels() {
        state.relabelled.forEach((button) => {
            if (button.dataset.mcThemeLabel !== undefined) {
                button.textContent = button.dataset.mcThemeLabel;
                delete button.dataset.mcThemeLabel;
            }
        });
        state.relabelled = [];
    }

    // -- moving things, and putting them back ------------------------------- //

    function move(node, parent, before, key) {
        if (!node || !parent) return false;
        if (node.parentElement === parent && (before ? node.nextSibling === before : parent.lastElementChild === node)) {
            return false;
        }
        if (!node.dataset.mcThemeMoved) {
            state.moved.push({node, parent: node.parentElement, next: node.nextSibling, key});
            node.dataset.mcThemeMoved = key;
        }
        parent.insertBefore(node, before || null);
        return true;
    }

    function prepend(node, parent, key) {
        if (!node || !parent) return false;
        if (node.parentElement === parent && parent.firstElementChild === node) return false;
        if (!node.dataset.mcThemeMoved) {
            state.moved.push({node, parent: node.parentElement, next: node.nextSibling, key});
            node.dataset.mcThemeMoved = key;
        }
        parent.insertBefore(node, parent.firstChild);
        return true;
    }

    function restoreMoves(keys) {
        const keep = [];
        state.moved.slice().reverse().forEach((record) => {
            if (keys && keys.indexOf(record.key) === -1) { keep.unshift(record); return; }
            const {node, parent, next} = record;
            if (parent) {
                const anchor = next && next.parentElement === parent ? next : null;
                parent.insertBefore(node, anchor);
            }
            delete node.dataset.mcThemeMoved;
        });
        state.moved = keep;
    }

    // -- the split view ---------------------------------------------------------- //

    function installSplit(config) {
        if (!config.split) { restoreMoves(["split"]); return; }
        ["txt2img", "img2img"].forEach((name) => {
            const settings = query("#" + name + "_settings");
            const toprow = query("#" + name + "_toprow");
            if (settings && toprow && !settings.contains(toprow)) prepend(toprow, settings, "split");
            const box = query("#" + name + "_generate_box");
            const container = query("#" + name + "_gallery_container");
            if (box && container) prepend(box, container, "split");
        });
    }

    // -- one scroller for the page --------------------------------------------- //
    //
    // Asked for: "the entire page takes one scroll action when dragging or
    // scroll. It should first move the left column until I reach bottom, then
    // the right column until I reach bottom. So both columns should never move
    // at the same time ... the generation button stays locked in position as I
    // scroll through the left column ... scrolling needs to now become a single
    // master control that controls the two columns from one master scroller."
    //
    // The master is the page's own scroller (the workspace's, under the Forge
    // Assistant's focus mode): a wheel, a finger, the scrollbar and the
    // keyboard all drive it, and the browser's compositor moves the columns.
    // Nothing here moves the page by the scroll (the momentum break below
    // only reads it, and acts once a fling crosses the boundary). The columns
    // are held by `position: sticky`, as the results column was held under
    // the bar before, from three numbers this measures and writes as custom
    // properties:
    //
    //  - The results column is held where it rests: a margin under the bar at
    //    least, but never above its place in the row, so Generate does not
    //    move at all (held at the bar's margin alone it moved up to it first).
    //  - The settings column is held once its bottom edge meets the view's: a
    //    sticky `top` of the view's height less the column's own, negative for
    //    a column taller than the view; a shorter one is held where the
    //    results are.
    //  - The row is given the least height the two phases need: the settings
    //    column's plus the results' part under the view. A sticky element
    //    never leaves its containing block, so the results column, held, is
    //    released exactly when the row's room below it is spent -- now the
    //    moment the settings reach their end -- and scrolls on to its own end
    //    with the settings held.
    //
    // Before this the row was as tall as the taller column: the results held
    // only while the settings were the taller, both moved together over the
    // last stretch (the infotext's height), and beside a short settings
    // column or a long infotext Generate scrolled off with the page.
    //
    // With the settings column docked in the Forge Assistant's panel the
    // column is out of the row (`position: fixed`, by its rule) and scrolls
    // inside the panel, which contains its own scrolling; nothing is written
    // for it, the row is the results' height, nothing holds, and the page
    // scrolls the results alone. Stacked on a phone, the results under the
    // settings, nothing is written either and the page scrolls as it does.
    //
    // The one price of the order: the generation details of the picture in
    // view are a whole settings column away. A Details button at the end of
    // the gallery's button row took the scroller there and back; the user
    // did not want it ("I don't need this scroll jump button"), so the row
    // is Forge's own and nothing here listens to the scroll.

    const SCROLL_ATTRIBUTE = "data-mc-theme-scroll";           // on the row: "settings-first"
    const SCROLL_SETTINGS_CLASS = "mc-theme-scroll-settings";   // the row's child that holds the settings
    const RESULTS_HOLD = "--mc-theme-results-hold";           // on the results column: its sticky top
    const SCROLL_HOLD = "--mc-theme-scroll-hold";             // on the row: the settings column's sticky top
    const SCROLL_ROW = "--mc-theme-scroll-row";               // on the row: its least height
    const SCROLL_TABS = ["txt2img", "img2img"];
    const DOCKED_CLASS = "forge-assistant-docked-column";     // the Forge Assistant's, on a column in its panel

    function px(value) {
        const found = parseFloat(value);
        return Number.isFinite(found) ? found : 0;
    }

    function laidOut(node) {
        if (!node || !node.getBoundingClientRect) return false;
        const box = node.getBoundingClientRect();
        return box.width > 0 && box.height > 0;
    }

    /** The nearest ancestor that scrolls up and down, or null for the page. */
    function scrollerOf(node) {
        let walk = node && node.parentElement;
        while (walk && walk !== document.body && walk !== document.documentElement) {
            const overflow = window.getComputedStyle(walk).overflowY || "visible";
            if (overflow === "auto" || overflow === "scroll" || overflow === "overlay") return walk;
            walk = walk.parentElement;
        }
        return null;
    }

    /** The child of `row` that holds `node`, or null if `row` does not. */
    function childOf(row, node) {
        let walk = node;
        while (walk && walk.parentElement !== row) walk = walk.parentElement;
        return walk || null;
    }

    /** The two numbers the stylesheet holds the columns by, in the scroller's
     *  own pixels. `settings` and `results` are the columns' heights, `view`
     *  the height from where the results column is held to the view's bottom
     *  edge, `hold` the sticky offset that holds it there. The settings
     *  column's offset holds it once its bottom edge meets the view's (a
     *  negative offset for a column taller than the view; a shorter one is
     *  held where the results are), and the row's least height is the
     *  settings column plus the results' part under the view: the room the
     *  results column has to stay held, spent exactly when the settings end.
     *  Pure, so the node tests hold it. */
    function scrollPlan(settings, results, view, hold) {
        const below = Math.max(0, results - view);
        return {top: Math.round(Math.min(hold, hold + view - settings)),
                minHeight: Math.round(settings + below)};
    }

    /** The scroll at which the first phase ends and the second begins: where
     *  the settings column's bottom edge (its margin box, as the sticky offset
     *  holds it) meets the view's bottom edge. `top` is the settings column's
     *  natural top in the scroller's own coordinates (the row's content edge,
     *  with the scroll added), `settings` its height with margins, `viewBottom`
     *  the view's bottom edge in the viewport's pixels. Null when the settings
     *  fit the view: then there is no first phase and nothing to break at.
     *  Pure, so the node tests hold it. */
    function scrollBoundary(top, settings, viewBottom) {
        const boundary = Math.round(top + settings - viewBottom);
        return boundary > 0 ? boundary : null;
    }

    /** A tab's row, the row's child that holds the settings (the column, or
     *  the accordion Forge wraps it in) and the results column; null without
     *  all three. */
    function findScrollRow(tabname) {
        const settings = query("#" + tabname + "_settings");
        const results = query("#" + tabname + "_results");
        if (!settings || !results || !results.parentElement) return null;
        const row = results.parentElement;
        const column = childOf(row, settings);
        if (!column || column === results) return null;
        return {row, column, results};
    }

    /** What scrolls the results column -- the page, or an ancestor that
     *  scrolls (the focus root) -- with its top edge and the view's bottom
     *  edge in the viewport's pixels, and how far it is scrolled. The bottom
     *  is inside the scroller's bottom padding, the page's own, so the
     *  settings rest at the page's end as the page's end does. */
    function viewOf(results) {
        const scroller = scrollerOf(results);
        if (scroller) {
            const box = scroller.getBoundingClientRect();
            const top = box.top + (scroller.clientTop || 0);
            return {scroller, top, scrolled: scroller.scrollTop || 0,
                    bottom: top + scroller.clientHeight - px(window.getComputedStyle(scroller).paddingBottom)};
        }
        const page = query(".gradio-container");
        const pad = page ? px(window.getComputedStyle(page).paddingBottom) : 0;
        return {scroller: null, top: 0, scrolled: window.scrollY || window.pageYOffset || 0,
                bottom: document.documentElement.clientHeight - pad};
    }

    /** The results column's sticky offset without this script's hold on it:
     *  the theme's margin under the bar. Read bare and put back in the same
     *  turn, with nothing painted in between. */
    function baseHold(results) {
        const mine = results.style.getPropertyValue(RESULTS_HOLD);
        if (mine) results.style.removeProperty(RESULTS_HOLD);
        const base = parseFloat(window.getComputedStyle(results).top);
        if (mine) results.style.setProperty(RESULTS_HOLD, mine);
        return base;
    }

    function clearScroll(entry) {
        const {row, column, results} = entry;
        [SCROLL_HOLD, SCROLL_ROW].forEach((name) => {
            if (row.style.getPropertyValue(name)) row.style.removeProperty(name);
        });
        if (results.style.getPropertyValue(RESULTS_HOLD)) results.style.removeProperty(RESULTS_HOLD);
        setClass(column, SCROLL_SETTINGS_CLASS, false);
        entry.boundary = null;
    }

    /** Measure one tab's row and write the numbers, or take them off; what
     *  was found, for `explainScroll`. Null for a tab that is not laid out,
     *  which keeps what it has: it is measured again when it is shown,
     *  because showing it resizes it. */
    function measureScroll(entry) {
        const {row, column, results} = entry;
        if (!laidOut(row) || !laidOut(results)) return null;
        const columnStyle = window.getComputedStyle(column);
        const docked = column.classList.contains(DOCKED_CLASS)
            || columnStyle.position === "fixed" || columnStyle.position === "absolute";
        const stacked = !docked
            && column.getBoundingClientRect().right > results.getBoundingClientRect().left + 1;
        let reason = docked ? "docked in the Forge Assistant's panel" : stacked ? "stacked under the settings" : "";
        if (!reason && window.getComputedStyle(results).position !== "sticky") {
            reason = "the results column is not held";
        }
        if (reason) { clearScroll(entry); return {reason}; }
        const view = viewOf(results);
        if (entry.scroller !== view.scroller) { entry.scroller = view.scroller; entry.shift = 0; }
        const rowStyle = window.getComputedStyle(row);
        const resultsStyle = window.getComputedStyle(results);
        // The results column's place in the row, in the viewport's pixels.
        const natural = row.getBoundingClientRect().top + px(rowStyle.paddingTop)
            + px(rowStyle.borderTopWidth) + px(resultsStyle.marginTop);
        // Held where it rests -- its place with the page at its start -- when
        // that is lower than the bar's margin, so that it never moves. Under
        // focus mode the assistant's script holds it at the workspace's inset
        // with an inline offset of its own, and this writes nothing.
        if (!view.scroller) {
            setVariable(results, RESULTS_HOLD, Math.round(Math.max(baseHold(results), natural + view.scrolled)) + "px");
        }
        const hold = parseFloat(window.getComputedStyle(results).top);
        if (!Number.isFinite(hold)) { clearScroll(entry); return {reason: "the results column is not held"}; }
        // Where it is held, in the viewport's pixels: the offset from the
        // scroller's edge, plus whatever the browser adds to it -- Chromium
        // measures a sticky offset from inside a scroller's padding, Firefox
        // from its edge -- read off the layout while the column is drawn
        // held, at that offset or below it, and remembered for a measure
        // taken while it is not: at rest (its place in the row lower than
        // the offset), or pushed up by the row's end in the second phase,
        // where the first build read the pushed position as the hold and
        // shrank the row at every measure the scroll met.
        const resultsBox = results.getBoundingClientRect();
        const columnBox = column.getBoundingClientRect();
        const expected = view.top + hold;
        const shifted = resultsBox.top > natural + 0.5;
        if (shifted && resultsBox.top >= expected - 0.5) entry.shift = resultsBox.top - expected;
        const heldAt = expected + (entry.shift || 0);
        const settingsHeight = columnBox.height + px(columnStyle.marginTop) + px(columnStyle.marginBottom);
        const resultsHeight = resultsBox.height + px(resultsStyle.marginTop) + px(resultsStyle.marginBottom);
        // No room under where the results are held -- a scroller with no
        // height for the moment (a phone's browser between full screen and
        // not, a tab in the middle of showing), a window shorter than the
        // bar's margin -- holds nothing and writes nothing: a plan for a
        // view below zero is a row of both columns' heights added up, and
        // the last plan was for a view that is gone. The next resize
        // measures again.
        const room = view.bottom - heldAt;
        if (!(room > 0)) { clearScroll(entry); return {reason: "the view has no room"}; }
        const plan = scrollPlan(settingsHeight, resultsHeight, room, hold);
        setClass(column, SCROLL_SETTINGS_CLASS, true);
        setVariable(row, SCROLL_HOLD, plan.top + "px");
        setVariable(row, SCROLL_ROW, plan.minHeight + "px");
        // The scroll at which the first phase ends, for the momentum break.
        const rowTop = row.getBoundingClientRect().top + px(rowStyle.paddingTop) + px(rowStyle.borderTopWidth);
        entry.boundary = scrollBoundary(rowTop + view.scrolled, settingsHeight, view.bottom);
        return {reason: "", scroller: view.scroller ? "workspace" : "page", hold, heldAt, shift: entry.shift || 0,
                settings: settingsHeight, results: resultsHeight, view: room,
                top: plan.top, minHeight: plan.minHeight, boundary: entry.boundary};
    }

    function measureScrollAll() {
        const scroll = state.scroll;
        if (!scroll) return;
        Object.keys(scroll.entries).forEach((tabname) => {
            measureScroll(scroll.entries[tabname]);
        });
    }

    /** Measure again on the next frame -- once, however many asked. */
    function scheduleScroll() {
        const scroll = state.scroll;
        if (!scroll || scroll.frame) return;
        scroll.frame = 1;
        const run = function () {
            scroll.frame = 0;
            attempt("order the page's scrolling", measureScrollAll);
        };
        if (typeof requestAnimationFrame === "function") requestAnimationFrame(run); else run();
    }

    function watchScroll(scroll, node) {
        if (!scroll.observer || scroll.observed.has(node)) return;
        scroll.observed.add(node);
        scroll.observer.observe(node);
    }

    function installScroll(config) {
        if (!config.scroll) { removeScroll(); return; }
        let scroll = state.scroll;
        if (!scroll) {
            scroll = state.scroll = {entries: {}, observer: null, observed: null, body: null, frame: 0};
            // Measured again when a column changes size -- the settings growing
            // (an accordion), the results (an infotext, the assistant's fill),
            // the row (its width: the dock, Forge's divider) -- and when the
            // body's classes change: focus mode coming and going changes what
            // scrolls the columns without resizing them.
            if (typeof ResizeObserver === "function") {
                scroll.observer = new ResizeObserver(function () { scheduleScroll(); });
                scroll.observed = new Set();
            }
            if (typeof MutationObserver === "function" && document.body) {
                scroll.body = new MutationObserver(function () { scheduleScroll(); });
                scroll.body.observe(document.body, {attributes: true, attributeFilter: ["class"]});
            }
        }
        installBrake(scroll);
        SCROLL_TABS.forEach((tabname) => {
            const found = findScrollRow(tabname);
            if (!found) return;
            const known = scroll.entries[tabname];
            if (known && (known.row !== found.row || known.column !== found.column
                          || known.results !== found.results)) {
                clearScroll(known);
                if (known.row.hasAttribute(SCROLL_ATTRIBUTE)) known.row.removeAttribute(SCROLL_ATTRIBUTE);
            }
            scroll.entries[tabname] = found;
            [found.row, found.column, found.results].forEach((node) => watchScroll(scroll, node));
            setAttr(found.row, SCROLL_ATTRIBUTE, "settings-first");
        });
        scheduleScroll();
    }

    function removeScroll() {
        const scroll = state.scroll;
        if (!scroll) return;
        removeBrake(scroll);
        if (scroll.observer) scroll.observer.disconnect();
        if (scroll.body) scroll.body.disconnect();
        if (scroll.frame && typeof cancelAnimationFrame === "function") cancelAnimationFrame(scroll.frame);
        Object.keys(scroll.entries).forEach((tabname) => {
            const entry = scroll.entries[tabname];
            clearScroll(entry);
            if (entry.row.hasAttribute(SCROLL_ATTRIBUTE)) entry.row.removeAttribute(SCROLL_ATTRIBUTE);
        });
        state.scroll = null;
    }

    /** What `measureScroll` sees, for the console: `mcTheme.explainScroll()`. */
    function explainScroll() {
        return SCROLL_TABS.map((tabname) => {
            const entry = state.scroll && state.scroll.entries[tabname];
            if (!entry) return {tab: tabname, reason: "no row"};
            return Object.assign({tab: tabname}, measureScroll(entry) || {reason: "not laid out"});
        });
    }

    // -- the momentum break between the two phases -------------------------------- //
    //
    // Asked for: "If I flick scroll to reach the bottom of column one, I do
    // not want it to continue momentum into column two. Instead it should
    // stop at the bottom like it hit an edge. From that bottom position, I
    // should be able to scroll to see what's above, or scroll to see what's
    // below." A finger's fling is the browser's: once the finger has lifted,
    // the compositor scrolls on by itself, and nothing a page says in a
    // listener reaches it in time -- a position written back from a scroll
    // event is one or two drawn frames late, and the fling keeps pushing.
    // What the browser does stop at is its own edge, so for the time of a
    // fling the page is given an edge at the boundary:
    //
    //  - Downward (the settings' phase into the results'): when the finger
    //    lifts with the scroll in the first phase, the scroller's range is
    //    cut to end exactly at the boundary -- `max-height` and
    //    `overflow-y: clip` on the row's ancestor just inside the scroller
    //    (Gradio's container on the page; under focus mode the element under
    //    the workspace), by the difference between the scroll height and
    //    the boundary plus the view. The row keeps its height, so the sticky
    //    columns are held exactly as before; only what lies below the view
    //    at the boundary is clipped, and nothing on screen changes. The
    //    fling meets the edge, the browser ends it there (Android draws its
    //    edge glow), and the clip comes off at the scroll's end.
    //  - Upward (the results' phase back into the settings'): a scroller's
    //    range cannot start above zero, so a fling read crossing the
    //    boundary upward is put back to it and the scroller is held
    //    unscrollable for the moment (`overflow: hidden`, as the Forge
    //    Assistant holds a page under a drag), which the browser reads as
    //    the end of the fling. The settings move for the frame or two the
    //    reading took and come back.
    //
    // The clip comes off at `scrollend`, the hold after fifteen frames without
    // a scroll event (the scroll back to the boundary has a `scrollend` of
    // its own, a frame later), and either at the next input of any kind (a
    // finger, a mouse, the wheel, a key) or the page going hidden, so the page
    // is whole again for the next gesture, whichever way it goes. A finger still on the screen is a drag, not a fling, and is
    // left alone; the wheel and the keyboard are not momentum and are never
    // stopped. Chromium synthesizes a fling from the finger's last movements
    // and stops it on the first scroll it cannot apply; measured in Chromium
    // (tests/test_theme_layout.py), not on a Gecko build.

    const BRAKE_QUIET_FRAMES = 15;   // about a quarter second at 60 Hz
    const BRAKE_CLIP_ATTRIBUTE = "data-mc-theme-scroll-clip";  // on the clipped ancestor while the clip is on
    const BRAKE_LOCK_ATTRIBUTE = "data-mc-theme-scroll-lock";  // on the scrolling element while it is held

    /** The entry whose row is on screen and has a boundary, or null. */
    function brakeEntry(scroll) {
        const names = Object.keys(scroll.entries);
        for (let i = 0; i < names.length; i += 1) {
            const entry = scroll.entries[names[i]];
            if (entry.boundary != null && laidOut(entry.row)) return entry;
        }
        return null;
    }

    /** The element that scrolls for an entry: the workspace, or the page's. */
    function scrollingElementOf(entry) {
        return entry.scroller || document.scrollingElement || document.documentElement;
    }

    /** The row's ancestor just inside the scroller, whose height ends the
     *  scroller's range: Gradio's container on the page (the app element
     *  above it is inline), the element under the workspace under focus
     *  mode; null where the row is the scroller's own child. */
    function clipTargetOf(entry) {
        if (!entry.scroller) {
            const container = query(".gradio-container");
            return container && container.contains(entry.row) ? container : null;
        }
        let walk = entry.row;
        while (walk && walk.parentElement && walk.parentElement !== entry.scroller) walk = walk.parentElement;
        return walk && walk !== entry.row && walk.parentElement === entry.scroller ? walk : null;
    }

    function saveInline(node, names) {
        const saved = {};
        names.forEach((name) => {
            saved[name] = {value: node.style.getPropertyValue(name), priority: node.style.getPropertyPriority(name)};
        });
        return saved;
    }

    function restoreInline(node, saved) {
        Object.keys(saved).forEach((name) => {
            if (saved[name].value) node.style.setProperty(name, saved[name].value, saved[name].priority);
            else node.style.removeProperty(name);
        });
    }

    /** Cut the scroller's range to end at the boundary, for a fling that
     *  starts in the first phase. Undone unless the range ends there. */
    function limitScroll(brake, entry) {
        const target = clipTargetOf(entry);
        if (!target || brake.clip) return false;
        const scroller = scrollingElementOf(entry);
        const excess = scroller.scrollHeight - (entry.boundary + scroller.clientHeight);
        if (!(excess > 0.5)) return false;
        const style = window.getComputedStyle(target);
        let height = target.getBoundingClientRect().height;
        if (style.boxSizing !== "border-box") {
            height -= px(style.paddingTop) + px(style.paddingBottom) + px(style.borderTopWidth) + px(style.borderBottomWidth);
        }
        const saved = saveInline(target, ["max-height", "overflow-y"]);
        target.style.setProperty("max-height", Math.max(0, height - excess) + "px", "important");
        target.style.setProperty("overflow-y", "clip", "important");
        if (Math.abs(scroller.scrollHeight - (entry.boundary + scroller.clientHeight)) > 1) {
            restoreInline(target, saved);
            return false;
        }
        setAttr(target, BRAKE_CLIP_ATTRIBUTE, "");
        brake.clip = {target, saved};
        return true;
    }

    /** Put a fling that crossed the boundary upward back to it and hold the
     *  scroller unscrollable for the moment, which ends the fling. */
    function lockScroll(brake, entry) {
        const scroller = scrollingElementOf(entry);
        if (entry.scroller) entry.scroller.scrollTop = entry.boundary;
        else window.scrollTo(window.scrollX || 0, entry.boundary);
        if (brake.lock) return true;
        const saved = saveInline(scroller, ["overflow", "scrollbar-gutter"]);
        scroller.style.setProperty("overflow", "hidden", "important");
        scroller.style.setProperty("scrollbar-gutter", "stable", "important");
        setAttr(scroller, BRAKE_LOCK_ATTRIBUTE, "");
        brake.lock = {target: scroller, saved};
        return true;
    }

    /** The page whole again: the clip and the hold off, the fling forgotten. */
    function releaseBrake(brake) {
        if (brake.frame && typeof cancelAnimationFrame === "function") cancelAnimationFrame(brake.frame);
        brake.frame = 0;
        brake.still = 0;
        brake.flinging = false;
        if (brake.clip) {
            restoreInline(brake.clip.target, brake.clip.saved);
            brake.clip.target.removeAttribute(BRAKE_CLIP_ATTRIBUTE);
            brake.clip = null;
        }
        if (brake.lock) {
            restoreInline(brake.lock.target, brake.lock.saved);
            brake.lock.target.removeAttribute(BRAKE_LOCK_ATTRIBUTE);
            brake.lock = null;
        }
    }

    function scrollOf(entry) {
        return entry.scroller ? entry.scroller.scrollTop : (window.scrollY || window.pageYOffset || 0);
    }

    function installBrake(scroll) {
        if (scroll.brake) return;
        const brake = scroll.brake = {fingers: 0, flinging: false, scrolled: false, clip: null, lock: null,
                                      frame: 0, still: 0, listeners: []};
        const listen = function (type, handler) {
            document.addEventListener(type, handler, {capture: true, passive: true});
            brake.listeners.push([type, handler]);
        };
        // Quiet is counted in animation frames, not on the clock: a frame's
        // scroll events are dispatched before its animation callbacks, so a
        // main thread starved for a while (four browsers on four cores in the
        // tests; a busy Gradio page) hands the fling's movements over before
        // the count can grow, where a timer could fire first and let the
        // break go before the crossing it was waiting for had been read.
        const quiet = function () {
            brake.still = 0;
            if (brake.frame) return;
            const tick = function () {
                brake.frame = 0;
                if (!brake.flinging) return;
                brake.still += 1;
                if (brake.still >= BRAKE_QUIET_FRAMES) { releaseBrake(brake); return; }
                brake.frame = requestAnimationFrame(tick);
            };
            brake.frame = requestAnimationFrame(tick);
        };
        listen("touchstart", function (event) {
            brake.fingers = event.touches ? event.touches.length : 1;
            releaseBrake(brake);
        });
        const lifted = function (event) {
            brake.fingers = event.touches ? event.touches.length : 0;
            if (brake.fingers) return;
            const entry = brakeEntry(scroll);
            if (!entry) return;
            brake.flinging = true;
            brake.scrolled = false;
            Object.keys(scroll.entries).forEach((name) => { scroll.entries[name].last = scrollOf(scroll.entries[name]); });
            if (scrollOf(entry) < entry.boundary - 1) limitScroll(brake, entry);
            quiet();
        };
        listen("touchend", lifted);
        listen("touchcancel", lifted);
        const input = function (event) {
            if (event.type === "pointerdown" && event.pointerType === "touch") return;
            releaseBrake(brake);
        };
        listen("pointerdown", input);
        listen("wheel", input);
        listen("keydown", input);
        listen("visibilitychange", input);
        listen("scroll", function (event) {
            const page = event.target === document || event.target === document.documentElement
                || event.target === document.body;
            Object.keys(scroll.entries).forEach((name) => {
                const entry = scroll.entries[name];
                if (page ? entry.scroller : entry.scroller !== event.target) return;
                const now = scrollOf(entry);
                const was = entry.last;
                entry.last = now;
                if (!brake.flinging || entry.boundary == null || was == null) return;
                if (now !== was) brake.scrolled = true;
                // Crossed upward: at or past the boundary a frame ago (a fling
                // can land on it exactly), below it now.
                if (was >= entry.boundary - 0.5 && now < entry.boundary - 0.5) lockScroll(brake, entry);
            });
            if (brake.flinging) quiet();
        });
        // A `scrollend` is the fling's end only once the fling has scrolled:
        // one from a scroll finished just before the finger lifted (a script's)
        // arrives a frame later, before the fling's first movement. And the
        // hold is let go on quiet alone: putting the scroll back to the
        // boundary is itself a scroll, whose own `scrollend` arrives a frame
        // later, while the fling is still alive and the hold not yet seen.
        listen("scrollend", function () {
            if (brake.flinging && brake.scrolled && !brake.lock) releaseBrake(brake);
        });
    }

    function removeBrake(scroll) {
        const brake = scroll.brake;
        if (!brake) return;
        releaseBrake(brake);
        brake.listeners.forEach(([type, handler]) => document.removeEventListener(type, handler, {capture: true}));
        scroll.brake = null;
    }

    /** The break's state, for the console and the tests: `mcTheme.explainBrake()`. */
    function explainBrake() {
        const brake = state.scroll && state.scroll.brake;
        if (!brake) return {installed: false};
        return {installed: true, fingers: brake.fingers, flinging: brake.flinging,
                clipped: !!brake.clip, locked: !!brake.lock};
    }

    // -- the quick settings drawer ------------------------------------------------ //

    function quickOpenState() {
        try { return localStorage.getItem(QUICK_STATE); } catch (error) { return null; }
    }

    function rememberQuick(open) {
        try { localStorage.setItem(QUICK_STATE, open ? "open" : "closed"); } catch (error) { /* ignore */ }
    }

    function setQuick(open) {
        const quick = state.quick;
        if (!quick) return;
        quick.hidden = !open;
        if (state.header) state.header.quickToggle.setAttribute("aria-expanded", open ? "true" : "false");
    }

    function toggleQuick() {
        const quick = state.quick;
        if (!quick) return;
        const open = quick.hidden;
        setQuick(open);
        rememberQuick(open);
    }

    function installQuick(config) {
        const settings = query("#quicksettings");
        if (config.quick === "off" || !settings) {
            removeQuick();
            return;
        }
        let quick = state.quick;
        if (!quick) {
            quick = make("aside", "mc-theme-drawer", {id: QUICK_ID, role: "complementary",
                                                       "aria-label": "Quick settings"});
            quick.hidden = true;
            mount(quick);
            state.quick = quick;
            const remembered = quickOpenState();
            const open = remembered ? remembered === "open" : config.quick === "open";
            setQuick(open);
        }
        quick.style.setProperty("--mc-theme-quick-width", config.quickWidth + "px");
        if (settings.parentElement !== quick) move(settings, quick, null, "quick");
    }

    function removeQuick() {
        restoreMoves(["quick"]);
        if (state.quick) {
            if (state.quick.parentElement) state.quick.parentElement.removeChild(state.quick);
            state.quick = null;
        }
    }

    // -- the extra networks drawer ------------------------------------------------------ //
    //
    // Forge Neo keeps the extra-network pages -- Textual Inversion, Checkpoints,
    // LoRA and the rest -- as tabs of `#<tab>_extra_tabs` beside Generation, so
    // a page shows in place of the generation UI. Here one page at a time is
    // drawn beside it instead: as a drawer over the page's right edge, or as
    // a column that takes its room from the generation UI (the layout
    // setting; the edge drags to a width either way, remembered per browser
    // until the width setting changes), under a head and a page strip of this
    // script's own, with the page's own search, sort and refresh in that head.
    //
    // The pages are never moved. Forge's script filters a page's cards with
    // `#<tab>_extra_tabs div.card` and shows a page's controls with
    // `#<tab>_extra_tabs .extra-networks-controls-div > div`: the first build
    // took the pages into an aside of its own outside that element, and from
    // then on the search box filtered nothing (the user's first report of
    // it). So everything this script adds -- the head, the strip, the edge --
    // is one more child of `#<tab>_extra_tabs`, the controls strip Forge
    // builds is moved only within it, and the pages stay Forge's own tab
    // panels in place, drawn where the stylesheet says. The head goes where
    // Forge's own strip was: before the pages. Forge renders a page twice --
    // empty when the UI is built, whole when the page loads -- and wires the
    // first render's controls into its strip, so every page keeps a second,
    // hidden copy of its controls under the same ids; Forge's sort reads the
    // pressed sort button with `gradioApp().querySelector`, the first match in
    // the document, and a head appended after the pages handed it the hidden
    // copy's button, which never changes: the sort buttons did nothing (the
    // user's report). Each tab has a drawer
    // of its own, which goes with its tab: nothing of it shows on a tab that
    // has no pages, and it is as it was on return. A page is taken only once
    // Forge's script has wired it (the strip it builds is the sign), which it
    // does well after the host says the page is loaded. The hidden tab strip
    // is marked by this script and hidden by the stylesheet under that mark
    // alone -- a tab bar is hidden by a script that knows it holds no
    // workspace, never by a selector that names tab bars.

    function currentTabName() {
        // The selected button's aria-controls names the panel (Gradio 4);
        // failing that, the one panel Gradio is displaying, as Forge's own
        // get_uiCurrentTabContent reads it.
        const strip = bar();
        const selected = strip && tabButtons(strip).find((button) => button.classList.contains("selected"));
        const controls = selected && selected.getAttribute("aria-controls");
        if (controls && controls.indexOf("tab_") === 0) return controls.slice(4);
        const panel = queryAll("#tabs > .tabitem[id^='tab_']").find(
            (candidate) => (candidate.getAttribute("style") || "").indexOf("display: none") === -1);
        return panel ? panel.id.slice(4) : null;
    }

    function setDisplay(element, value) {
        if (element.style.display !== value) element.style.display = value;
    }

    function setVariable(element, name, value) {
        if (element.style.getPropertyValue(name) !== value) element.style.setProperty(name, value);
    }

    const NETWORKS_HOST_CLASS = "mc-theme-networks-host";
    const NETWORKS_OPEN_CLASS = "mc-theme-networks-open";
    const NETWORKS_PAGE_CLASS = "mc-theme-networks-page";
    const NETWORKS_LAYOUT_ATTRIBUTE = "data-mc-theme-networks";
    const NETWORKS_WIDTH_STATE = "mc-theme-networks-width";   // {width, setting}, per browser
    const NETWORKS_STEP = 10;                                 // the keyboard's step at the edge

    function networksOpenState() {
        try { return localStorage.getItem(NETWORKS_STATE); } catch (error) { return null; }
    }

    function rememberNetworks(open) {
        try { localStorage.setItem(NETWORKS_STATE, open ? "open" : "closed"); } catch (error) { /* ignore */ }
    }

    function clampNetworksWidth(width) {
        return Math.min(NETWORKS_WIDTH_MAX, Math.max(NETWORKS_WIDTH_MIN, Math.round(width)));
    }

    /** The width the edge was dragged to, with the setting it was dragged
     *  from: a changed setting wins over it. */
    function rememberedWidth() {
        try {
            const parsed = JSON.parse(localStorage.getItem(NETWORKS_WIDTH_STATE) || "null");
            if (!parsed || typeof parsed !== "object") return null;
            const width = Number(parsed.width);
            if (!Number.isFinite(width)) return null;
            return {width: clampNetworksWidth(width), setting: Number(parsed.setting)};
        } catch (error) {
            return null;
        }
    }

    function rememberWidth(width, setting) {
        try {
            localStorage.setItem(NETWORKS_WIDTH_STATE, JSON.stringify({width, setting}));
        } catch (error) { /* ignore */ }
    }

    function eachHost(fn) {
        const networks = state.networks;
        if (!networks) return;
        Object.keys(networks.hosts).forEach((tabname) => fn(networks.hosts[tabname], tabname));
    }

    function setNetworks(open) {
        const networks = state.networks;
        if (!networks) return;
        networks.open = !!open;
        eachHost((entry) => setClass(entry.host, NETWORKS_OPEN_CLASS, networks.open));
        if (state.header) setAttr(state.header.networksToggle, "aria-expanded", networks.open ? "true" : "false");
    }

    function toggleNetworks() {
        const networks = state.networks;
        if (!networks) return;
        setNetworks(!networks.open);
        rememberNetworks(networks.open);
    }

    function applyNetworksWidth(width) {
        const networks = state.networks;
        if (!networks) return;
        networks.width = width;
        eachHost((entry) => setVariable(entry.host, "--mc-theme-networks-width", width + "px"));
    }

    function showNetworksPage(tabname, pageId) {
        const networks = state.networks;
        const entry = networks && networks.hosts[tabname];
        if (!entry) return;
        entry.page = pageId;
        entry.pages.forEach((page) => setClass(page, NETWORKS_SHOWN_CLASS, page.id === pageId));
        Array.prototype.slice.call(entry.strip.children).forEach((button) => {
            const active = button.dataset.mcThemePage === pageId;
            setClass(button, "mc-theme-networks-active", active);
            setAttr(button, "aria-selected", active ? "true" : "false");
        });
        // The page's own controls -- search, sort, refresh -- which Forge's
        // script gathered into one strip; the same rule as its own
        // extraNetworksShowControlsForPage, applied where the strip is now.
        if (entry.controls) {
            Array.prototype.slice.call(entry.controls.children).forEach((child) => {
                setDisplay(child, child.id === pageId + "_controls" ? "" : "none");
            });
        }
        sortShown(pageId);
    }

    // The cards containers this script has had Forge sort. A rendering of a
    // page's cards is a new container (Forge's load and its Refresh replace
    // the page's HTML), so a container is sorted once for each rendering.
    const sortedCards = typeof WeakSet === "function" ? new WeakSet() : null;

    /** Forge filters and sorts a page when its own tab is selected
     *  (`applyExtraNetworkFilter`, from the tab's select event), and its
     *  load renders the cards in the listing's order; the drawer shows a
     *  page without selecting Forge's tab, so a page opened newest created
     *  first showed the listing's order until a search or a press. Asked
     *  once for each rendering of a page's cards: Forge's sort puts the
     *  cards back into their container, and asked on every update that
     *  would raise the next update itself. */
    function sortShown(pageId) {
        if (!sortedCards || typeof applyExtraNetworkFilter !== "function") return;
        const cards = document.getElementById(pageId + "_cards");
        if (!cards || sortedCards.has(cards)) return;
        sortedCards.add(cards);
        applyExtraNetworkFilter(pageId);
    }

    /** The header's button follows the tab: a tab with no pages has no
     *  drawer to open. The drawers themselves go with their tabs. */
    function syncNetworks() {
        const networks = state.networks;
        if (!networks || !state.header) return;
        const entry = networks.hosts[currentTabName()];
        const any = !!(entry && entry.pages.length);
        const toggle = state.header.networksToggle;
        if (toggle.disabled !== !any) toggle.disabled = !any;
        if (any) setAttr(toggle, "aria-controls", entry.chrome.id);
        else if (toggle.hasAttribute("aria-controls")) toggle.removeAttribute("aria-controls");
    }

    /** The drawer's edge: a drag, or the arrow keys, to a width within the
     *  setting's bounds, remembered with the setting it was dragged from. */
    function wireHandle(handle, tabname) {
        let drag = null;
        const end = function (event) {
            if (!drag || (event && event.pointerId !== undefined && event.pointerId !== drag.pointerId)) return;
            drag = null;
            if (document.body) document.body.classList.remove("mc-theme-networks-resizing");
            const networks = state.networks;
            if (networks) rememberWidth(networks.width, networks.setting);
        };
        handle.addEventListener("pointerdown", function (event) {
            if (event.button !== undefined && event.button !== 0) return;
            const entry = state.networks && state.networks.hosts[tabname];
            if (!entry) return;
            event.preventDefault();
            // The edge keeps its distance from the pointer: a drag of a
            // hundred pixels is a hundred pixels, with no jump at the grab.
            const box = entry.chrome.getBoundingClientRect();
            drag = {pointerId: event.pointerId, right: box.right, offset: event.clientX - box.left};
            if (document.body) document.body.classList.add("mc-theme-networks-resizing");
            try { handle.setPointerCapture(event.pointerId); } catch (error) { /* an optimisation */ }
        });
        handle.addEventListener("pointermove", function (event) {
            if (!drag || event.pointerId !== drag.pointerId) return;
            applyNetworksWidth(clampNetworksWidth(drag.right - (event.clientX - drag.offset)));
        });
        handle.addEventListener("pointerup", end);
        handle.addEventListener("pointercancel", end);
        handle.addEventListener("lostpointercapture", function () { end(null); });
        handle.addEventListener("keydown", function (event) {
            const networks = state.networks;
            if (!networks) return;
            let width = null;
            if (event.key === "ArrowLeft") width = networks.width + NETWORKS_STEP;
            else if (event.key === "ArrowRight") width = networks.width - NETWORKS_STEP;
            if (width === null) return;
            event.preventDefault();
            applyNetworksWidth(clampNetworksWidth(width));
            rememberWidth(networks.width, networks.setting);
        });
    }

    function installNetworks(config) {
        if (config.networks === "off") { removeNetworks(); return; }
        let networks = state.networks;
        if (!networks) {
            networks = state.networks = {hosts: {}, open: false, width: null, setting: null, mode: null};
            const remembered = networksOpenState();
            networks.open = remembered ? remembered === "open" : config.networks === "open";
        }
        if (networks.setting !== config.networksWidth) {
            // The setting, first read or since changed, wins over a width
            // dragged from an earlier setting.
            networks.setting = config.networksWidth;
            const kept = rememberedWidth();
            networks.width = kept && kept.setting === config.networksWidth ? kept.width : config.networksWidth;
        }
        networks.mode = config.networksMode;

        ["txt2img", "img2img"].forEach((tabname) => {
            const host = query("#" + tabname + "_extra_tabs");
            if (!host) return;
            const children = Array.prototype.slice.call(host.children);
            const nav = children.find(
                (child) => child.classList.contains("tab-nav") || child.getAttribute("role") === "tablist");
            const pages = children.filter((child) => child.classList.contains("extra-page"));
            if (!nav || !pages.length) return;
            // Forge wires a page's search, sort and refresh after the page's
            // HTML has arrived, well after the host says it is loaded, and it
            // looks for the pages as children of its own tabs. The strip it
            // gathers their controls into is the sign that it has been: a
            // page is taken only once that strip exists, so nothing is moved
            // from under Forge's own script. Once taken, the strip is the
            // drawer's (a later apply finds it there, not in the nav).
            let entry = networks.hosts[tabname];
            const gathered = (entry && entry.controls)
                || nav.querySelector(":scope > .extra-networks-controls-div");
            if (!gathered) return;
            if (!entry) {
                const chrome = make("div", "mc-theme-networks",
                                    {id: NETWORKS_ID + "-" + tabname, role: "complementary",
                                     "aria-label": "Extra networks", "data-mc-theme-tab": tabname});
                const head = make("div", "mc-theme-networks-head");
                const title = make("div", "mc-theme-networks-title");
                title.textContent = "Extra networks";
                const close = action("mc-theme-networks-close", "x", "Close");
                close.addEventListener("click", function () { setNetworks(false); rememberNetworks(false); });
                head.appendChild(title);
                head.appendChild(close);
                const strip = make("div", "mc-theme-networks-strip", {role: "tablist"});
                const handle = make("div", "mc-theme-networks-handle",
                                    {role: "separator", "aria-orientation": "vertical", tabindex: "0",
                                     "aria-label": "Resize the extra networks", title: "Drag to resize"});
                wireHandle(handle, tabname);
                chrome.appendChild(head);
                chrome.appendChild(strip);
                chrome.appendChild(handle);
                // Inside Forge's own element, where its selectors reach, and
                // before the pages, where Forge's strip was (`placeHead`).
                placeHead(host, nav, chrome);
                entry = networks.hosts[tabname] = {host, nav, chrome, strip, controls: null, pages: [],
                                                   page: null, observer: null};
                if (typeof ResizeObserver === "function") {
                    // The page is drawn under the head, wherever the head's
                    // height -- the strip wrapping, the controls' row -- puts
                    // its bottom edge.
                    entry.observer = new ResizeObserver(function () {
                        const height = Math.ceil(chrome.getBoundingClientRect().height);
                        setVariable(host, "--mc-theme-networks-chrome", height + "px");
                    });
                    entry.observer.observe(chrome);
                }
            }
            // Put back before the pages if anything moved it after one --
            // and left alone where it is, which is every apply but the first.
            placeHead(host, nav, entry.chrome);
            setClass(host, NETWORKS_HOST_CLASS, true);
            setClass(host, NETWORKS_OPEN_CLASS, networks.open);
            setAttr(host, NETWORKS_LAYOUT_ATTRIBUTE, config.networksMode);
            setVariable(host, "--mc-theme-networks-width", networks.width + "px");
            setVariable(host, "--mc-theme-networks-card", config.networksCard + "px");
            setClass(nav, NETWORKS_NAV_CLASS, true);
            if (entry.controls !== gathered) {
                move(gathered, entry.chrome, null, "networks");
                entry.controls = gathered;
            }
            const buttons = tabButtons(nav);
            pages.forEach((page) => {
                setClass(page, NETWORKS_PAGE_CLASS, true);
                if (entry.pages.indexOf(page) === -1) {
                    entry.pages.push(page);
                    const button = make("button", "mc-theme-networks-tab", {type: "button", role: "tab"});
                    button.textContent = pageLabel(page, pages, buttons);
                    button.dataset.mcThemeTab = tabname;
                    button.dataset.mcThemePage = page.id;
                    button.addEventListener("click", function () { showNetworksPage(tabname, page.id); });
                    entry.strip.appendChild(button);
                }
            });
            showNetworksPage(tabname, entry.page || entry.pages[0].id);
        });
        if (state.header) setAttr(state.header.networksToggle, "aria-expanded", networks.open ? "true" : "false");
        syncNetworks();
    }

    /** The drawer's head right after Forge's tab strip, ahead of every page,
     *  where Forge's own controls strip sits in document order. Moved only
     *  when it is not there already: a move is a change to a child list, and
     *  one on every apply would raise Forge's next after-update itself. */
    function placeHead(host, nav, chrome) {
        if (chrome.parentElement === host && chrome.previousElementSibling === nav) return;
        host.insertBefore(chrome, nav.nextSibling);
    }

    /** The name Forge's tab button gives a page: by aria-controls where
     *  Gradio writes it, else by place (the buttons before the pages' are
     *  Generation's), else the page's own id. */
    function pageLabel(page, pages, buttons) {
        const named = buttons.find((button) => button.getAttribute("aria-controls") === page.id);
        if (named) return named.textContent.trim();
        const byPlace = buttons[buttons.length - pages.length + pages.indexOf(page)];
        if (byPlace && !byPlace.getAttribute("aria-controls")) return byPlace.textContent.trim();
        return page.id.replace(/^(txt2img|img2img)_/, "").replace(/_/g, " ");
    }

    function removeNetworks() {
        const networks = state.networks;
        restoreMoves(["networks"]);
        if (networks) {
            eachHost((entry) => {
                if (entry.observer) entry.observer.disconnect();
                setClass(entry.nav, NETWORKS_NAV_CLASS, false);
                entry.pages.forEach((page) => {
                    setClass(page, NETWORKS_SHOWN_CLASS, false);
                    setClass(page, NETWORKS_PAGE_CLASS, false);
                });
                setClass(entry.host, NETWORKS_HOST_CLASS, false);
                setClass(entry.host, NETWORKS_OPEN_CLASS, false);
                if (entry.host.hasAttribute(NETWORKS_LAYOUT_ATTRIBUTE)) entry.host.removeAttribute(NETWORKS_LAYOUT_ATTRIBUTE);
                ["--mc-theme-networks-width", "--mc-theme-networks-card", "--mc-theme-networks-chrome"]
                    .forEach((name) => entry.host.style.removeProperty(name));
                // Generation is the selected tab, where Forge shows no page's controls.
                if (entry.controls) {
                    Array.prototype.slice.call(entry.controls.children).forEach((child) => setDisplay(child, "none"));
                }
                if (entry.chrome.parentElement) entry.chrome.parentElement.removeChild(entry.chrome);
            });
            state.networks = null;
        }
        if (state.header) setAttr(state.header.networksToggle, "aria-expanded", "false");
    }

    // -- the generation info, as a panel ---------------------------------------------------- //
    //
    // Forge writes the parameters of the last generation under the gallery as
    // one paragraph: the prompt, "Negative prompt: …" and a last line of
    // "Steps: 8, Sampler: …, Seed: …" pairs, joined by line breaks. The panel
    // draws the same text as three things -- the prompt and the negative
    // prompt each in a block with a copy button, the pairs as chips -- beside
    // Forge's paragraph, which the stylesheet then keeps out of view. Nothing
    // is rewritten: the paragraph is still what Forge's own buttons read.

    const PAIR = /\s*([\w\s\-\/]+):\s*("(?:\\.|[^\\"])+"|[^,]*)(?:,|$)/g;
    const NEGATIVE = "Negative prompt:";

    function unquote(value) {
        const text = String(value).trim();
        if (text.length >= 2 && text.charAt(0) === '"' && text.charAt(text.length - 1) === '"') {
            try { return String(JSON.parse(text)); } catch (error) { return text.slice(1, -1); }
        }
        return text;
    }

    function pairsOf(line) {
        const pairs = [];
        PAIR.lastIndex = 0;
        let match;
        while ((match = PAIR.exec(line)) !== null) {
            if (match.index === PAIR.lastIndex) PAIR.lastIndex += 1;
            const key = match[1].trim();
            const value = unquote(match[2]);
            if (key) pairs.push([key, value]);
            if (match[0] === "") break;
        }
        return pairs;
    }

    /** {prompt, negative, pairs} from an infotext, or null for nothing. */
    function parseInfotext(text) {
        const lines = String(text || "").replace(/\r/g, "").split("\n")
            .map((line) => line.trim()).filter((line) => line.length);
        if (!lines.length) return null;
        let pairs = [];
        let last = lines[lines.length - 1];
        const found = pairsOf(last);
        let body = lines;
        if (found.length >= 3) { pairs = found; body = lines.slice(0, -1); }
        const prompt = [];
        const negative = [];
        let inNegative = false;
        body.forEach((line) => {
            let piece = line;
            if (piece.indexOf(NEGATIVE) === 0) { inNegative = true; piece = piece.slice(NEGATIVE.length).trim(); }
            (inNegative ? negative : prompt).push(piece);
        });
        return {prompt: prompt.join("\n"), negative: negative.join("\n"), pairs};
    }

    function infotextOf(element) {
        // The paragraph's lines, through the breaks Forge joins them with (a
        // <br> and a newline each, so the newline after a break is the break's).
        const html = element.innerHTML || "";
        const holder = document.createElement("div");
        holder.innerHTML = html.replace(/<br\s*\/?>\s*/gi, "\n");
        return holder.textContent || "";
    }

    function copyText(text, button) {
        const done = function () {
            button.classList.add("mc-theme-info-copied");
            setTimeout(function () { button.classList.remove("mc-theme-info-copied"); }, 1200);
        };
        try {
            if (navigator.clipboard && navigator.clipboard.writeText) {
                navigator.clipboard.writeText(text).then(done, function () { fallbackCopy(text); done(); });
                return;
            }
        } catch (error) { /* fall through */ }
        fallbackCopy(text);
        done();
    }

    function fallbackCopy(text) {
        try {
            const area = document.createElement("textarea");
            area.value = text;
            area.setAttribute("readonly", "");
            area.style.position = "fixed";
            area.style.opacity = "0";
            document.body.appendChild(area);
            area.select();
            document.execCommand("copy");
            document.body.removeChild(area);
        } catch (error) { /* nothing to be done */ }
    }

    function infoBlock(label, text) {
        const block = make("div", "mc-theme-info-block");
        const head = make("div", "mc-theme-info-label");
        head.textContent = label;
        const copy = make("button", "mc-theme-info-copy", {type: "button", title: "Copy " + label.toLowerCase(),
                                                           "aria-label": "Copy " + label.toLowerCase()});
        copy.innerHTML = GLYPH["clipboard-list"];
        copy.addEventListener("click", function () { copyText(text, copy); });
        head.appendChild(copy);
        const body = make("div", "mc-theme-info-text");
        body.textContent = text;
        block.appendChild(head);
        block.appendChild(body);
        return block;
    }

    function renderInfo(parsed, text) {
        const panel = make("div", INFO_CLASS);
        if (parsed.prompt) panel.appendChild(infoBlock("Prompt", parsed.prompt));
        if (parsed.negative) panel.appendChild(infoBlock("Negative prompt", parsed.negative));
        if (parsed.pairs.length) {
            const chips = make("div", "mc-theme-info-params");
            parsed.pairs.forEach(([key, value]) => {
                const chip = make("span", "mc-theme-info-chip");
                const name = make("b");
                name.textContent = key;
                chip.appendChild(name);
                chip.appendChild(document.createTextNode(" " + value));
                chip.title = key + ": " + value;
                chips.appendChild(chip);
            });
            const all = make("button", "mc-theme-info-copy mc-theme-info-copy-all",
                             {type: "button", title: "Copy everything", "aria-label": "Copy everything"});
            all.innerHTML = GLYPH["clipboard-list"];
            all.addEventListener("click", function () { copyText(text.trim(), all); });
            chips.appendChild(all);
            panel.appendChild(chips);
        }
        return panel;
    }

    function installInfo(config) {
        if (!config.info) { removeInfo(); return; }
        queryAll(".infotext[id^='html_info_']").forEach((source) => {
            // Asked on every update of the page; the paragraph's markup says
            // whether there is anything new to draw.
            const html = source.innerHTML || "";
            const known = state.infoPanels.get(source);
            if (known && known.html === html) return;
            const text = infotextOf(source);
            const parsed = parseInfotext(text);
            if (known && known.panel.parentElement) known.panel.parentElement.removeChild(known.panel);
            if (!parsed || (!parsed.prompt && !parsed.negative && !parsed.pairs.length)) {
                source.classList.remove(INFO_SOURCE_CLASS);
                state.infoPanels.delete(source);
                return;
            }
            const panel = renderInfo(parsed, text);
            source.parentElement.insertBefore(panel, source.nextSibling);
            source.classList.add(INFO_SOURCE_CLASS);
            state.infoPanels.set(source, {panel, html});
        });
    }

    function removeInfo() {
        state.infoPanels.forEach((entry, source) => {
            if (entry.panel.parentElement) entry.panel.parentElement.removeChild(entry.panel);
            source.classList.remove(INFO_SOURCE_CLASS);
        });
        state.infoPanels.clear();
    }

    // -- the prompt boxes ----------------------------------------------------------- //

    function installPrompts(config) {
        const html = root();
        if (config.prompts) {
            if (html.getAttribute(PROMPTS_ATTRIBUTE) !== "resizable") html.setAttribute(PROMPTS_ATTRIBUTE, "resizable");
        } else if (html.hasAttribute(PROMPTS_ATTRIBUTE)) {
            html.removeAttribute(PROMPTS_ATTRIBUTE);
        }
    }

    // -- the icons --------------------------------------------------------------------- //

    function installIcons(config) {
        if (!config.icons) { restoreIcons(); return; }
        queryAll("button").forEach((button) => {
            if (button.dataset.mcThemeIcon !== undefined) return;
            if (button.closest && button.closest(".mc-theme-actions")) return;
            const text = button.textContent;
            const glyph = iconFor(text);
            if (!glyph) return;
            button.dataset.mcThemeIcon = plain(text);
            button.dataset.mcThemeText = text;
            button.innerHTML = glyph;
            if (!button.getAttribute("aria-label")) {
                const label = button.getAttribute("title") || button.getAttribute("data-tooltip") || plain(text);
                button.setAttribute("aria-label", label);
            }
            state.swapped.push(button);
        });
    }

    function restoreIcons() {
        state.swapped.forEach((button) => {
            if (button.dataset.mcThemeIcon === undefined) return;
            button.textContent = button.dataset.mcThemeText || button.dataset.mcThemeIcon;
            delete button.dataset.mcThemeIcon;
            delete button.dataset.mcThemeText;
        });
        state.swapped = [];
    }

    // -- leaving the page, the typeface ---------------------------------------------- //

    function installConfirm(config) {
        if (config.confirm && !state.unload) {
            state.unload = function (event) { event.preventDefault(); event.returnValue = ""; return ""; };
            window.addEventListener("beforeunload", state.unload);
        } else if (!config.confirm && state.unload) {
            window.removeEventListener("beforeunload", state.unload);
            state.unload = null;
        }
    }

    function fontUrl(family) {
        return "https://fonts.googleapis.com/css2?family=" + encodeURIComponent(family).replace(/%20/g, "+")
             + ":wght@400;500;600&display=swap";
    }

    function installFont(config) {
        if (!config.font || !document.head) return;
        let link = document.getElementById(FONT_LINK_ID);
        const href = fontUrl(config.font);
        if (link && link.getAttribute("href") === href) return;
        if (!link) {
            link = make("link", "", {id: FONT_LINK_ID, rel: "stylesheet"});
            document.head.appendChild(link);
        }
        link.setAttribute("href", href);
    }

    // -- putting it together -------------------------------------------------------------- //

    function teardown() {
        removeScroll();
        removeNetworks();
        removeInfo();
        restoreMoves();
        restoreIcons();
        restoreLabels();
        removeHeader();
        if (state.quick) {
            if (state.quick.parentElement) state.quick.parentElement.removeChild(state.quick);
            state.quick = null;
        }
        if (root().hasAttribute(PROMPTS_ATTRIBUTE)) root().removeAttribute(PROMPTS_ATTRIBUTE);
        installConfirm({confirm: false});
    }

    function mark(value) {
        const html = root();
        if (!value) { if (html.hasAttribute(ATTRIBUTE)) html.removeAttribute(ATTRIBUTE); return; }
        if (html.getAttribute(ATTRIBUTE) !== value) html.setAttribute(ATTRIBUTE, value);
    }

    function apply() {
        const config = read();
        remember(config);
        if (!config.enabled) { teardown(); mark(null); return; }
        if (lobeRoot()) {
            teardown();
            mark("lobe");
            if (!state.lastLobe) console.info("Model Chain theme: the Lobe theme is on this page; standing down");
            state.lastLobe = true;
            return;
        }
        state.lastLobe = false;
        mark("on");
        const strip = bar();
        if (strip) {
            attempt("install the header", function () { installHeader(strip, config); });
            attempt("name the tabs", function () { installLabels(strip, config); });
        }
        attempt("lay out the split view", function () { installSplit(config); });
        attempt("order the page's scrolling", function () { installScroll(config); });
        attempt("install the quick settings drawer", function () { installQuick(config); });
        attempt("install the extra networks drawer", function () { installNetworks(config); });
        attempt("draw the generation info", function () { installInfo(config); });
        attempt("mark the prompt boxes", function () { installPrompts(config); });
        attempt("draw the icons", function () { installIcons(config); });
        attempt("ask before leaving", function () { installConfirm(config); });
        attempt("load the typeface", function () { installFont(config); });
        scheduleFit();
    }

    // -- a generation in the results column ---------------------------------------- //
    //
    // While Forge's progress bar runs, its Interrupt and Skip row is not drawn
    // (style.css, "a generation in the results column"). A press on the bar
    // brings the row back for the rest of the run, and another puts it away:
    // the mark is on the bar, which Forge removes when the run ends, so the
    // next run starts without the row again. The keyboard needs none of this
    // -- Forge's Escape and Alt+Enter press the buttons wherever they are --
    // but a touchscreen has no Escape.

    const SHOW_CONTROLS = "mc-theme-show-controls";
    const RUNNING_BARS = "[id$='_results_panel'] > .progressDiv:not(.mc-progress-flash)";

    // A press counts on the bar or in what it paints above itself -- the ooze
    // rises 34 px over a bar 14 px tall, and a finger needs more than the bar
    // -- by where it lands rather than by an element there: the room above the
    // bar is empty, and a pseudo-element on the host's bar is a WebUI theme's
    // (tests/test_progress.py holds that). Never a press on a control.
    function pressBar(event) {
        if (!event || root().getAttribute(ATTRIBUTE) !== "on") return false;
        const target = event.target;
        if (target && target.closest && target.closest("button, a, input, textarea, select, label")) {
            return false;
        }
        const x = event.clientX;
        const y = event.clientY;
        const bars = document.querySelectorAll(RUNNING_BARS);
        for (let index = 0; index < bars.length; index += 1) {
            const bar = bars[index];
            const box = bar.getBoundingClientRect();
            if (!(box.width > 0 && box.height > 0)) continue;
            let top = box.top;
            Array.prototype.forEach.call(bar.children, (child) => {
                if (window.getComputedStyle(child).position !== "absolute") return;
                top = Math.min(top, child.getBoundingClientRect().top);
            });
            if (x >= box.left && x <= box.right && y >= top && y <= box.bottom) {
                bar.classList.toggle(SHOW_CONTROLS);
                return true;
            }
        }
        return false;
    }

    function wire() {
        if (state.wired) return;
        state.wired = true;
        document.addEventListener("click", function (event) {
            attempt("show or hide Interrupt and Skip", function () { pressBar(event); });
        });
        document.addEventListener("click", function (event) {
            if (!state.header || state.header.menu.hidden) return;
            if (state.header.actions.contains(event.target)) return;
            closeMenu();
        });
        document.addEventListener("keydown", function (event) {
            if (event.key !== "Escape") return;
            if (state.header && !state.header.menu.hidden) { closeMenu(); return; }
            if (state.quick && !state.quick.hidden) { setQuick(false); rememberQuick(false); return; }
            if (state.networks && state.networks.open) { setNetworks(false); rememberNetworks(false); }
        });
        window.addEventListener("resize", scheduleFit);
        window.addEventListener("resize", scheduleScroll);
        // What else changes what the columns are held against, with no
        // resize of the window to say so: a pinch, which changes the
        // visible viewport; the page shown again, from the browser's cache
        // or from behind another tab, with whatever the browser did to it
        // meanwhile; the browser's full screen coming or going, which focus
        // mode asks for on a phone. Each is measured again, once, on the
        // next frame -- a page that is hidden draws no frame until shown.
        if (window.visualViewport && typeof window.visualViewport.addEventListener === "function") {
            window.visualViewport.addEventListener("resize", scheduleScroll);
        }
        window.addEventListener("pageshow", scheduleScroll);
        document.addEventListener("visibilitychange", scheduleScroll);
        ["fullscreenchange", "webkitfullscreenchange"].forEach(function (type) {
            document.addEventListener(type, scheduleScroll);
        });
    }

    // The first paint: the last known configuration marks the root at once,
    // so a page that was themed yesterday is themed before the host says it
    // is loaded, and corrected the moment the settings arrive.
    attempt("mark the root from the last configuration", function () {
        const last = recall();
        if (last && last.enabled) {
            mark("on");
            if (last.prompts) root().setAttribute(PROMPTS_ATTRIBUTE, "resizable");
        }
    });
    wire();
    if (typeof onUiLoaded === "function") onUiLoaded(apply);
    if (typeof onOptionsChanged === "function") onOptionsChanged(apply);
    if (typeof onAfterUiUpdate === "function") onAfterUiUpdate(apply);
    if (typeof onUiTabChange === "function") onUiTabChange(function () { scheduleFit(); scheduleScroll(); attempt("follow the tab", syncNetworks); });

    window.mcTheme = {
        OPTION, DEFAULTS, ATTRIBUTE, PROMPTS_ATTRIBUTE, OVERFLOW_CLASS, QUICK_ID, NETWORKS_ID,
        normalise, spaced, plan, iconFor, plain, nextThemeUrl, fontUrl, apply, fit, lobeRoot,
        parseInfotext, showNetworksPage, toggleNetworks, setNetworks, pressBar, SHOW_CONTROLS,
        NETWORKS_LAYOUTS,
        scrollPlan, scrollBoundary, explainScroll, explainBrake, SCROLL_ATTRIBUTE, SCROLL_SETTINGS_CLASS,
        RESULTS_HOLD, SCROLL_HOLD, SCROLL_ROW, BRAKE_CLIP_ATTRIBUTE, BRAKE_LOCK_ATTRIBUTE, BRAKE_QUIET_FRAMES,
        icons: Object.keys(ICONS),
        glyphs: Object.keys(GLYPH),
    };
})();
