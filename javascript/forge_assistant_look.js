// Forge Assistant -- the launcher's look, and the dialog that edits it.
//
// Collapsed, the launcher is the one control on the page that is always
// there, and it became the way to every workspace. It should be possible to
// find it at a glance, and to make it one's own. So it can be restyled: fill,
// text, border, corners, size, a glow, and the same animations the progress
// bar can wear -- including one that simply follows whatever the progress
// bar has been set to.
//
// Three rules shape it.
//
// *Nothing moves on the main thread.* Every animation here is `transform` or
// `opacity` on a layer of its own, so the compositor runs it without the page
// having to lay anything out or paint anything, and it stops the moment the
// launcher is not displayed -- which is whenever the panel is open.
//
// *Nothing is spent until it is asked for.* The dialog is built the first
// time it is opened, and the decorations an animation needs are added to the
// launcher only while that animation is chosen.
//
// *What is stored is checked on the way back in.* The look lives in
// `localStorage`, which the user, an extension or a bad week can put anything
// into. Every field is validated against what it can be; anything that fails
// is the default instead, and nothing read from storage ever reaches the page
// as markup or as an unchecked CSS value.

(function () {
    "use strict";

    const NS = (window.forgeAssistant = window.forgeAssistant || {});

    const VERSION = 1;
    const MAX_TITLE = 40;

    const MOTIONS = ["none", "match", "sheen", "pulse", "neon", "ooze", "aurora"];
    const MOTION_LABELS = {
        none: "None", match: "Match progress bar", sheen: "Sheen", pulse: "Pulse",
        neon: "Neon", ooze: "Ooze", aurora: "Aurora",
    };
    // What each animation is made of. Neon is sheen and pulse at once, the
    // way the progress bar's Neon is its sheen and glow at once.
    const PARTS = {
        none: [], sheen: ["sheen"], pulse: ["pulse"], neon: ["sheen", "pulse", "lit"],
        ooze: ["ooze"], aurora: ["aurora"],
    };
    const SPEEDS = {slow: 1.6, normal: 1, fast: 0.6};
    const SIZES = ["compact", "regular", "large"];
    const GLOWS = ["off", "soft", "strong"];
    const ICONS = ["", "●", "✦", "⚡", "✨", "🎨", "🤖",
                   "💬", "🧪", "🔥", "★"];
    const SWATCHES = ["#111827", "#ffffff", "#ef4444", "#f97316", "#f59e0b", "#22c55e",
                      "#14b8a6", "#3b82f6", "#6366f1", "#a855f7", "#ec4899", "#0b0f1a"];
    const BUBBLES = 7;

    const DEFAULTS = Object.freeze({
        version: VERSION,
        title: "",          // "" is the label from Settings
        icon: "",           // "" is whatever Settings says about the icon
        size: "regular",
        text: "auto",       // "auto" picks black or white against the fill
        fill: "",           // "" is the theme's own background
        fill2: "",          // a second stop makes it a gradient
        angle: 135,
        stroke: "",         // "" is the theme's own border colour
        strokeWidth: 1,
        radius: 10,         // 999 is a pill
        glow: "off",
        accent: "",         // the colour effects are drawn in; "" follows the rest
        motion: "none",
        speed: "normal",
        anyway: false,      // animate even when the system asks for less motion
    });

    // A look is a complete set of choices, so a preset is one too -- applying
    // one replaces everything, and fine-tuning starts from there.
    const PRESETS = [
        {id: "classic", name: "Classic", look: {}},
        {id: "neon", name: "Neon", look: {
            fill: "#0b0f1a", text: "#7df9ff", stroke: "#00e5ff", strokeWidth: 2,
            radius: 999, glow: "strong", accent: "#00e5ff", motion: "neon"}},
        {id: "toxic", name: "Toxic", look: {
            fill: "#0d2b0f", fill2: "#1f6b1a", angle: 180, text: "auto",
            stroke: "#7cff3a", strokeWidth: 2, radius: 14, glow: "soft",
            accent: "#66ff33", motion: "ooze", icon: "🧪"}},
        {id: "aurora", name: "Aurora", look: {
            fill: "#111827", text: "#ffffff", strokeWidth: 2, radius: 999,
            motion: "aurora", icon: "✨"}},
        {id: "sunset", name: "Sunset", look: {
            fill: "#c2410c", fill2: "#be185d", angle: 135, text: "#ffffff",
            strokeWidth: 0, radius: 999, glow: "soft", accent: "#f97316", motion: "sheen"}},
        {id: "ember", name: "Ember", look: {
            fill: "#1a0a00", fill2: "#7a1f00", angle: 160, text: "#ffcf7a",
            stroke: "#ff7a18", strokeWidth: 2, radius: 12, glow: "strong",
            accent: "#ff7a18", motion: "pulse", icon: "🔥"}},
        {id: "mono", name: "Mono", look: {
            fill: "#ffffff", text: "#111111", stroke: "#111111", strokeWidth: 2,
            radius: 0, motion: "none"}},
        {id: "match", name: "Match progress bar", look: {
            strokeWidth: 1, radius: 999, glow: "soft", motion: "match"}},
    ];

    // -- validation ---------------------------------------------------------- //

    const HEX = /^#[0-9a-f]{6}$/i;

    function colour(value, allowAuto) {
        const text = String(value === undefined || value === null ? "" : value).trim();
        if (allowAuto && text === "auto") return "auto";
        return HEX.test(text) ? text.toLowerCase() : "";
    }

    function number(value, low, high, fallback) {
        const found = Number(value);
        if (!Number.isFinite(found)) return fallback;
        return Math.min(high, Math.max(low, Math.round(found)));
    }

    function oneOf(value, allowed, fallback) {
        return allowed.indexOf(value) >= 0 ? value : fallback;
    }

    // Printable text only: control characters are dropped, and the length is
    // capped where the launcher would start truncating it anyway.
    function title(value) {
        const text = String(value === undefined || value === null ? "" : value)
            .replace(/[\u0000-\u001f\u007f-\u009f\u2028\u2029]/g, "")
            .trim();
        return Array.from(text).slice(0, MAX_TITLE).join("");
    }

    /** Anything in, a complete and valid look out. */
    function sanitize(raw) {
        const found = raw && typeof raw === "object" ? raw : {};
        const radius = Number(found.radius) >= 999 ? 999
            : number(found.radius, 0, 28, DEFAULTS.radius);
        return {
            version: VERSION,
            title: title(found.title),
            icon: oneOf(found.icon, ICONS, DEFAULTS.icon),
            size: oneOf(found.size, SIZES, DEFAULTS.size),
            text: colour(found.text, true) || "auto",
            fill: colour(found.fill),
            fill2: colour(found.fill2),
            angle: number(found.angle, 0, 360, DEFAULTS.angle),
            stroke: colour(found.stroke),
            strokeWidth: number(found.strokeWidth, 0, 4, DEFAULTS.strokeWidth),
            radius,
            glow: oneOf(found.glow, GLOWS, DEFAULTS.glow),
            accent: colour(found.accent),
            motion: oneOf(found.motion, MOTIONS, DEFAULTS.motion),
            speed: oneOf(found.speed, Object.keys(SPEEDS), DEFAULTS.speed),
            anyway: found.anyway === true,
        };
    }

    function withPreset(id) {
        const preset = PRESETS.find((item) => item.id === id) || PRESETS[0];
        return sanitize(Object.assign({}, DEFAULTS, preset.look));
    }

    // -- colour arithmetic ----------------------------------------------------- //

    function channels(hex) {
        const value = parseInt(hex.slice(1), 16);
        return [(value >> 16) & 255, (value >> 8) & 255, value & 255];
    }

    function luminance(hex) {
        const [r, g, b] = channels(hex).map((channel) => {
            const c = channel / 255;
            return c <= 0.03928 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4);
        });
        return 0.2126 * r + 0.7152 * g + 0.0722 * b;
    }

    /** WCAG contrast ratio between two #rrggbb colours: 1 to 21. */
    function contrast(first, second) {
        const a = luminance(first);
        const b = luminance(second);
        return (Math.max(a, b) + 0.05) / (Math.min(a, b) + 0.05);
    }

    // The colour effects are drawn in, when it is a colour this file can do
    // arithmetic on: chosen, or the most colourful thing already chosen. A
    // look that follows the progress bar has its colour in a CSS variable,
    // which is only known to the page.
    function accentOf(look) {
        if (look.accent) return look.accent;
        if (look.motion === "match") return "";
        return look.stroke || look.fill2 || look.fill || "";
    }

    // What the title is actually drawn on. Ooze paints its sludge over the
    // fill, in the effect colour, so for Ooze that is the background -- a
    // title chosen against a dark fill is unreadable on bright sludge, which
    // is how the first Toxic preset came out.
    function backdrop(look) {
        if (look.motion === "ooze" && accentOf(look)) return [accentOf(look)];
        return [look.fill, look.fill2].filter(Boolean);
    }

    // Black or white, whichever reads better against what is behind the title
    // -- against the worse of two stops when there are two, because the title
    // crosses the whole gradient.
    function autoText(look) {
        const stops = backdrop(look);
        if (!stops.length) return "";
        const worst = (candidate) => Math.min.apply(null, stops.map((stop) =>
            contrast(candidate, stop)));
        return worst("#000000") >= worst("#ffffff") ? "#000000" : "#ffffff";
    }

    /** The legibility of the title, or null when the theme decides both. */
    function legibility(look) {
        const text = look.text === "auto" ? autoText(look) : look.text;
        const stops = backdrop(look);
        if (!text || !stops.length) return null;
        return Math.min.apply(null, stops.map((stop) => contrast(text, stop)));
    }

    // -- the progress bar -------------------------------------------------------- //

    function option(name, fallback) {
        try {
            if (typeof opts === "undefined" || opts === null) return fallback;
            const value = opts[name];
            return value === undefined || value === null ? fallback : value;
        } catch (error) {
            return fallback;
        }
    }

    // Which of these animations the progress bar is wearing. Read from the
    // same settings `model_chain_progress.js` reads, at the moment the look is
    // applied, so a change in Settings is followed the next time it is.
    function progressMotion() {
        if (!option("model_chain_style_enable", false)) return "none";
        const theme = String(option("model_chain_style_theme", "Flat"));
        if (theme === "Custom") {
            if (option("model_chain_style_sheen", false)) {
                return option("model_chain_style_glow", false) ? "neon" : "sheen";
            }
            return option("model_chain_style_glow", false) ? "pulse" : "none";
        }
        return {Sheen: "sheen", Pulse: "pulse", Neon: "neon", Ooze: "ooze"}[theme] || "none";
    }

    // -- resolving a look to what the launcher carries ---------------------------- //

    /** Everything `apply` writes, worked out without touching the page.
     *
     * `settings` is the Settings page's label, icon and appearance -- what a
     * look with nothing chosen falls back to.
     */
    function resolve(look, settings) {
        const found = sanitize(look);
        const base = settings || {};
        const motion = found.motion === "match" ? progressMotion() : found.motion;
        const vars = {};
        if (found.fill) {
            vars["--fa-look-fill"] = found.fill2
                ? "linear-gradient(" + found.angle + "deg, " + found.fill + ", " + found.fill2 + ")"
                : found.fill;
            vars["--fa-look-base"] = found.fill;
        }
        const text = found.text === "auto" ? autoText(found) : found.text;
        if (text) vars["--fa-look-text"] = text;
        if (found.stroke) vars["--fa-look-stroke"] = found.stroke;
        vars["--fa-look-stroke-width"] = found.strokeWidth + "px";
        vars["--fa-look-radius"] = found.radius + "px";
        vars["--fa-look-speed"] = String(SPEEDS[found.speed]);
        // The colour effects are drawn in: chosen, or the progress bar's own
        // when following it, or the most colourful thing already chosen.
        const accent = accentOf(found);
        vars["--fa-look-accent"] = accent
            || (found.motion === "match" ? "var(--mc-progress-fill, var(--color-accent, #3b82f6))"
                : "var(--color-accent, #3b82f6)");
        return {
            vars,
            motion,
            parts: PARTS[motion] || [],
            size: found.size,
            glow: found.glow,
            anyway: found.anyway,
            label: found.title || base.label || "Forge Assistant",
            // An icon chosen here is shown whatever Settings says; one not
            // chosen leaves Settings in charge of whether there is one.
            icon: found.icon,
            showIcon: found.icon ? true : base.appearance !== "text",
            showLabel: base.appearance !== "icon" || !!found.title,
        };
    }

    // -- applying ---------------------------------------------------------------- //

    const VAR_NAMES = ["--fa-look-fill", "--fa-look-base", "--fa-look-text", "--fa-look-stroke",
                       "--fa-look-stroke-width", "--fa-look-radius", "--fa-look-speed",
                       "--fa-look-accent"];

    function make(tag, className) {
        const node = document.createElement(tag);
        if (className) node.className = className;
        return node;
    }

    // One bubble, placed and timed at random, the way the progress bar seeds
    // its own: a field of identical bubbles on one period reads as a pattern.
    function bubble() {
        const node = make("span", "mc-ooze-bubble forge-assistant-look-bubble");
        const size = 3 + Math.random() * 5;
        const duration = 1.6 + Math.random() * 1.6;
        node.style.setProperty("--x", (8 + Math.random() * 84).toFixed(1) + "%");
        node.style.setProperty("--size", size.toFixed(1) + "px");
        node.style.setProperty("--dur", duration.toFixed(2) + "s");
        node.style.setProperty("--delay", (-Math.random() * duration).toFixed(2) + "s");
        node.style.setProperty("--lift", (14 + Math.random() * 16).toFixed(0) + "px");
        node.style.setProperty("--drift", ((Math.random() - 0.5) * 12).toFixed(1) + "px");
        return node;
    }

    // The layers an animation draws on. Behind the title (z-index -1 inside
    // the launcher's own stacking context) and never a target for a press.
    // Built only for the animation chosen, and taken away when it is not.
    function decorate(node, parts) {
        let clip = node.querySelector(":scope > .forge-assistant-look-clip");
        let air = node.querySelector(":scope > .forge-assistant-look-air");
        const needsClip = parts.some((part) => part === "sheen" || part === "ooze"
            || part === "aurora");
        const needsAir = parts.some((part) => part === "pulse" || part === "ooze");
        if (needsClip && !clip) {
            clip = make("span", "forge-assistant-look-clip");
            clip.setAttribute("aria-hidden", "true");
            node.insertBefore(clip, node.firstChild);
        } else if (!needsClip && clip) {
            clip.remove();
        }
        if (needsAir && !air) {
            air = make("span", "forge-assistant-look-air");
            air.setAttribute("aria-hidden", "true");
            node.insertBefore(air, node.firstChild);
        } else if (!needsAir && air) {
            air.remove();
            air = null;
        }
        if (!air) return;
        const halo = air.querySelector(".forge-assistant-look-halo");
        if (parts.indexOf("pulse") >= 0 && !halo) {
            air.appendChild(make("span", "forge-assistant-look-halo"));
        } else if (parts.indexOf("pulse") < 0 && halo) {
            halo.remove();
        }
        const bubbles = air.querySelectorAll(".forge-assistant-look-bubble");
        if (parts.indexOf("ooze") >= 0 && !bubbles.length) {
            for (let i = 0; i < BUBBLES; i += 1) air.appendChild(bubble());
        } else if (parts.indexOf("ooze") < 0) {
            Array.prototype.forEach.call(bubbles, (item) => item.remove());
        }
    }

    /** Put a look on a launcher-shaped element: the real one, or a preview. */
    function apply(node, look, settings, parts) {
        if (!node) return null;
        const found = resolve(look, settings);
        VAR_NAMES.forEach((name) => node.style.removeProperty(name));
        Object.keys(found.vars).forEach((name) => node.style.setProperty(name, found.vars[name]));
        node.setAttribute("data-look", "");
        node.setAttribute("data-look-size", found.size);
        node.setAttribute("data-look-glow", found.glow);
        node.setAttribute("data-look-fill", found.vars["--fa-look-fill"] ? "set" : "theme");
        if (found.parts.length) {
            node.setAttribute("data-look-motion", found.parts.join(" "));
        } else {
            node.removeAttribute("data-look-motion");
        }
        if (found.anyway) node.setAttribute("data-look-anyway", "");
        else node.removeAttribute("data-look-anyway");
        decorate(node, found.parts);
        const label = parts && parts.label;
        const icon = parts && parts.icon;
        if (label) {
            label.textContent = found.label;
            label.hidden = !found.showLabel;
        }
        if (icon) {
            if (found.icon) icon.textContent = found.icon;
            icon.hidden = !found.showIcon;
        }
        return found;
    }

    // -- storage ------------------------------------------------------------------ //

    function key() {
        return "forge-assistant-look:" + ((NS.basePath && NS.basePath()) || "/");
    }

    function load() {
        let raw = null;
        try {
            raw = window.localStorage.getItem(key());
        } catch (error) {
            raw = null;
        }
        if (!raw) return sanitize(DEFAULTS);
        try {
            const found = JSON.parse(raw);
            if (!found || found.version !== VERSION) return sanitize(DEFAULTS);
            return sanitize(found);
        } catch (error) {
            return sanitize(DEFAULTS);
        }
    }

    function save(look) {
        try {
            window.localStorage.setItem(key(), JSON.stringify(sanitize(look)));
            return true;
        } catch (error) {
            return false;
        }
    }

    // -- the dialog ----------------------------------------------------------------- //
    //
    // Built once, the first time it is opened, into the assistant's root. A
    // native <dialog> opened modally: the browser supplies the backdrop, the
    // focus containment and the top layer (above focus mode and above another
    // extension's dialog layer), and Escape closes it as a cancel.
    //
    // Every control changes a draft. The draft is drawn on the preview at the
    // top *and* on the real launcher behind the backdrop, so what you see is
    // what you get. Cancel, Escape and a press on the backdrop put the look
    // back as it was; Save keeps it.

    function Editor(shell) {
        this.shell = shell;
        this.dialog = null;
        this.before = null;
        this.draft = null;
        this.controls = {};
    }

    Editor.prototype.isOpen = function () {
        return !!(this.dialog && this.dialog.open);
    };

    Editor.prototype.settings = function () {
        const settings = this.shell.settings || {};
        return {label: settings.label, appearance: settings.appearance};
    };

    Editor.prototype.open = function () {
        if (!this.dialog) this.build();
        this.before = load();
        this.draft = sanitize(this.before);
        this.sync();
        const dialog = this.dialog;
        try {
            if (typeof dialog.showModal === "function") dialog.showModal();
            else dialog.setAttribute("open", "");
        } catch (error) {
            dialog.setAttribute("open", "");
        }
        // The first control, not the dialog: a keyboard lands somewhere useful.
        const first = this.dialog.querySelector(".forge-assistant-look-preset");
        if (first && first.focus) first.focus();
        return true;
    };

    Editor.prototype.close = function (keep) {
        if (!this.dialog) return;
        if (keep) {
            save(this.draft);
        } else if (this.before) {
            this.shell.applyLook(this.before);
        }
        if (this.dialog.open && typeof this.dialog.close === "function") this.dialog.close();
        else this.dialog.removeAttribute("open");
        this.before = null;
        const launcher = this.shell.nodes && this.shell.nodes.launcher;
        if (launcher && !launcher.hidden && launcher.focus) launcher.focus();
    };

    // One change to the draft, drawn everywhere it shows.
    Editor.prototype.set = function (changes) {
        this.draft = sanitize(Object.assign({}, this.draft, changes));
        this.sync();
    };

    Editor.prototype.sync = function () {
        const draft = this.draft;
        const c = this.controls;
        apply(c.preview, draft, this.settings(), {label: c.previewLabel, icon: c.previewIcon});
        this.shell.applyLook(draft);
        if (c.title.value !== draft.title) c.title.value = draft.title;
        c.title.placeholder = this.settings().label || "Forge Assistant";
        const pressed = (group, value) => {
            (c[group] || []).forEach((button) => {
                button.setAttribute("aria-pressed", String(button.dataset.value === String(value)));
            });
        };
        pressed("icons", draft.icon);
        pressed("sizes", draft.size);
        pressed("text", draft.text);
        pressed("fill", draft.fill);
        pressed("fill2", draft.fill2);
        pressed("stroke", draft.stroke);
        pressed("strokeWidths", draft.strokeWidth);
        pressed("radii", draft.radius === 999 ? "pill" : "custom");
        pressed("glows", draft.glow);
        pressed("accent", draft.accent);
        pressed("motions", draft.motion);
        pressed("speeds", draft.speed);
        c.radius.value = String(draft.radius === 999 ? 28 : draft.radius);
        c.radiusValue.textContent = draft.radius === 999 ? "Pill" : draft.radius + "px";
        c.angle.value = String(draft.angle);
        c.angleValue.textContent = draft.angle + "°";
        c.angleRow.hidden = !draft.fill2;
        c.anyway.checked = draft.anyway;
        c.anywayRow.hidden = !(reducedMotion() && draft.motion !== "none");
        c.matchNote.hidden = draft.motion !== "match";
        if (draft.motion === "match") {
            const following = progressMotion();
            c.matchNote.textContent = following === "none"
                ? "The progress bar has no animation set in Settings, so this has none either."
                : "Following the progress bar: " + MOTION_LABELS[following] + ".";
        }
        const ratio = legibility(draft);
        c.warning.hidden = !(ratio !== null && ratio < 3);
        if (ratio !== null && ratio < 3) {
            c.warning.textContent = "The title is hard to read on this fill ("
                + ratio.toFixed(1) + ":1). Try Auto for the text colour.";
        }
    };

    function reducedMotion() {
        try {
            return !!(window.matchMedia
                && window.matchMedia("(prefers-reduced-motion: reduce)").matches);
        } catch (error) {
            return false;
        }
    }

    Editor.prototype.build = function () {
        const shell = this.shell;
        const c = this.controls;
        const dialog = make("dialog", "forge-assistant-look-dialog");
        dialog.setAttribute("aria-labelledby", "forge-assistant-look-heading");
        const sheet = make("div", "forge-assistant-look-sheet");
        dialog.appendChild(sheet);

        // The heading and the preview stay on screen while the rest scrolls:
        // the whole point is watching the button change.
        const top = make("div", "forge-assistant-look-top");
        const heading = make("h2", "forge-assistant-look-heading");
        heading.id = "forge-assistant-look-heading";
        heading.textContent = "Customize the button";
        const dismiss = make("button", "forge-assistant-icon-button");
        dismiss.type = "button";
        dismiss.textContent = "✕";
        dismiss.setAttribute("aria-label", "Close without saving");
        dismiss.addEventListener("click", () => this.close(false));
        const bar = make("div", "forge-assistant-look-bar");
        bar.appendChild(heading);
        bar.appendChild(dismiss);
        top.appendChild(bar);
        const stage = make("div", "forge-assistant-look-stage");
        const preview = make("span", "forge-assistant-launcher forge-assistant-look-preview");
        preview.setAttribute("aria-hidden", "true");
        const previewIcon = make("span", "forge-assistant-launcher-icon");
        previewIcon.textContent = "●";
        const previewLabel = make("span", "forge-assistant-launcher-label");
        preview.appendChild(previewIcon);
        preview.appendChild(previewLabel);
        stage.appendChild(preview);
        top.appendChild(stage);
        sheet.appendChild(top);
        Object.assign(c, {preview, previewIcon, previewLabel});

        const body = make("div", "forge-assistant-look-body");
        sheet.appendChild(body);

        const section = (name) => {
            const block = make("section", "forge-assistant-look-section");
            const label = make("h3", "forge-assistant-look-label");
            label.textContent = name;
            block.appendChild(label);
            body.appendChild(block);
            return block;
        };
        const row = (parent, name) => {
            const line = make("div", "forge-assistant-look-row");
            if (name) {
                const label = make("span", "forge-assistant-look-caption");
                label.textContent = name;
                line.appendChild(label);
            }
            const group = make("div", "forge-assistant-look-group");
            line.appendChild(group);
            parent.appendChild(line);
            return {line, group};
        };
        const chip = (group, text, value, onPick, className) => {
            const button = make("button", className || "forge-assistant-look-chip");
            button.type = "button";
            button.textContent = text;
            button.dataset.value = String(value);
            button.setAttribute("aria-pressed", "false");
            button.addEventListener("click", () => onPick(value));
            group.appendChild(button);
            return button;
        };
        // A row of colours: the theme's own, twelve chosen ones, and the
        // system picker for anything else. Each swatch is a 44px target.
        const palette = (parent, name, field, options) => {
            const {group} = row(parent, name);
            const buttons = [];
            const extra = (options && options.first) || [];
            extra.forEach(([text, value]) => {
                buttons.push(chip(group, text, value, (picked) => this.set({[field]: picked})));
            });
            SWATCHES.forEach((hex) => {
                const swatch = chip(group, "", hex, (picked) => this.set({[field]: picked}),
                                    "forge-assistant-look-swatch");
                swatch.style.setProperty("--swatch", hex);
                swatch.setAttribute("aria-label", hex);
                swatch.title = hex;
                buttons.push(swatch);
            });
            const custom = make("label", "forge-assistant-look-custom");
            const picker = make("input");
            picker.type = "color";
            picker.setAttribute("aria-label", name + ": any colour");
            picker.addEventListener("input", () => this.set({[field]: picker.value}));
            custom.appendChild(picker);
            const text = make("span");
            text.textContent = "Any";
            custom.appendChild(text);
            group.appendChild(custom);
            c[field] = buttons;
            return group;
        };

        // Presets first: one press to something that looks finished.
        const presets = section("Presets");
        const {group: presetGroup} = row(presets);
        presetGroup.classList.add("forge-assistant-look-presets");
        PRESETS.forEach((preset) => {
            const card = make("button", "forge-assistant-look-preset");
            card.type = "button";
            card.setAttribute("aria-label", preset.name + " preset");
            const mini = make("span", "forge-assistant-launcher forge-assistant-look-preview");
            mini.setAttribute("aria-hidden", "true");
            const miniLabel = make("span", "forge-assistant-launcher-label");
            const miniIcon = make("span", "forge-assistant-launcher-icon");
            mini.appendChild(miniIcon);
            mini.appendChild(miniLabel);
            card.appendChild(mini);
            const name = make("span", "forge-assistant-look-preset-name");
            name.textContent = preset.name;
            card.appendChild(name);
            card.addEventListener("click", () => {
                // The title is the user's own words, not part of a style.
                const kept = this.draft.title;
                this.draft = withPreset(preset.id);
                this.draft.title = kept;
                this.sync();
            });
            presetGroup.appendChild(card);
            apply(mini, withPreset(preset.id), {label: "Aa", appearance: "text"},
                  {label: miniLabel, icon: miniIcon});
        });

        const words = section("Title and icon");
        const {group: titleGroup} = row(words);
        const input = make("input", "forge-assistant-look-title");
        input.type = "text";
        input.maxLength = MAX_TITLE;
        input.setAttribute("aria-label", "Title on the button");
        input.addEventListener("input", () => this.set({title: input.value}));
        titleGroup.appendChild(input);
        c.title = input;
        const {group: iconGroup} = row(words, "Icon");
        c.icons = ICONS.map((icon) => chip(iconGroup, icon || "None", icon,
                                           (picked) => this.set({icon: picked})));
        const {group: sizeGroup} = row(words, "Size");
        c.sizes = SIZES.map((size) => chip(sizeGroup, size[0].toUpperCase() + size.slice(1),
                                           size, (picked) => this.set({size: picked})));
        palette(words, "Text colour", "text", {first: [["Auto", "auto"]]});

        const fill = section("Fill");
        palette(fill, "Colour", "fill", {first: [["Theme", ""]]});
        palette(fill, "Blend into", "fill2", {first: [["None", ""]]});
        const angle = row(fill, "Direction");
        const angleInput = make("input", "forge-assistant-look-range");
        angleInput.type = "range";
        angleInput.min = "0";
        angleInput.max = "360";
        angleInput.step = "15";
        angleInput.setAttribute("aria-label", "Gradient direction");
        angleInput.addEventListener("input", () => this.set({angle: angleInput.value}));
        const angleValue = make("span", "forge-assistant-look-value");
        angle.group.appendChild(angleInput);
        angle.group.appendChild(angleValue);
        Object.assign(c, {angle: angleInput, angleValue, angleRow: angle.line});

        const edge = section("Border and corners");
        palette(edge, "Border colour", "stroke", {first: [["Theme", ""]]});
        const {group: widthGroup} = row(edge, "Border width");
        c.strokeWidths = [0, 1, 2, 3, 4].map((width) => chip(widthGroup, width ? width + "px" : "None",
                                                             width, (picked) => this.set({strokeWidth: picked})));
        const corners = row(edge, "Corners");
        const radiusInput = make("input", "forge-assistant-look-range");
        radiusInput.type = "range";
        radiusInput.min = "0";
        radiusInput.max = "28";
        radiusInput.step = "1";
        radiusInput.setAttribute("aria-label", "Corner radius");
        radiusInput.addEventListener("input", () => this.set({radius: radiusInput.value}));
        const radiusValue = make("span", "forge-assistant-look-value");
        corners.group.appendChild(radiusInput);
        corners.group.appendChild(radiusValue);
        c.radii = [chip(corners.group, "Pill", "pill", () => this.set({radius: 999}))];
        Object.assign(c, {radius: radiusInput, radiusValue});

        const motion = section("Effects");
        const {group: motionGroup} = row(motion, "Animation");
        c.motions = MOTIONS.map((name) => chip(motionGroup, MOTION_LABELS[name], name,
                                               (picked) => this.set({motion: picked})));
        const note = make("p", "forge-assistant-look-note");
        note.hidden = true;
        motion.appendChild(note);
        c.matchNote = note;
        const {group: speedGroup} = row(motion, "Speed");
        c.speeds = Object.keys(SPEEDS).map((speed) => chip(speedGroup,
            speed[0].toUpperCase() + speed.slice(1), speed, (picked) => this.set({speed: picked})));
        const {group: glowGroup} = row(motion, "Glow");
        c.glows = GLOWS.map((glow) => chip(glowGroup, glow[0].toUpperCase() + glow.slice(1), glow,
                                           (picked) => this.set({glow: picked})));
        palette(motion, "Effect colour", "accent", {first: [["Auto", ""]]});
        const anyway = row(motion);
        const toggle = make("label", "forge-assistant-look-toggle");
        const box = make("input");
        box.type = "checkbox";
        box.addEventListener("change", () => this.set({anyway: box.checked}));
        toggle.appendChild(box);
        const toggleText = make("span");
        toggleText.textContent = "Your system asks for reduced motion. Animate anyway";
        toggle.appendChild(toggleText);
        anyway.group.appendChild(toggle);
        Object.assign(c, {anyway: box, anywayRow: anyway.line});

        const warning = make("p", "forge-assistant-look-warning");
        warning.setAttribute("role", "status");
        warning.hidden = true;
        body.appendChild(warning);
        c.warning = warning;

        const foot = make("div", "forge-assistant-look-foot");
        const reset = make("button", "forge-assistant-look-action");
        reset.type = "button";
        reset.textContent = "Reset";
        reset.title = "Back to the look the button had before it was customized";
        reset.addEventListener("click", () => {
            this.draft = sanitize(DEFAULTS);
            this.sync();
        });
        const cancel = make("button", "forge-assistant-look-action");
        cancel.type = "button";
        cancel.textContent = "Cancel";
        cancel.addEventListener("click", () => this.close(false));
        const keep = make("button", "forge-assistant-look-action forge-assistant-look-save");
        keep.type = "button";
        keep.textContent = "Save";
        keep.addEventListener("click", () => this.close(true));
        foot.appendChild(reset);
        foot.appendChild(cancel);
        foot.appendChild(keep);
        sheet.appendChild(foot);

        // Escape is the browser's `cancel`; it is taken over only so it can
        // put the look back before the dialog goes.
        dialog.addEventListener("cancel", (event) => {
            event.preventDefault();
            this.close(false);
        });
        // A press on the backdrop is a press on the dialog element itself,
        // outside the sheet: that is a cancel too.
        dialog.addEventListener("click", (event) => {
            if (event.target === dialog) this.close(false);
        });
        (shell.nodes.root || document.body).appendChild(dialog);
        this.dialog = dialog;
    };

    NS.look = {
        VERSION, DEFAULTS, PRESETS, MOTIONS, SPEEDS, SIZES, GLOWS, ICONS, SWATCHES,
        sanitize, resolve, apply, load, save, contrast, autoText, legibility, accentOf,
        progressMotion, withPreset, key, Editor,
        editor(shell) {
            if (!shell.lookEditor) shell.lookEditor = new Editor(shell);
            return shell.lookEditor;
        },
    };
})();
