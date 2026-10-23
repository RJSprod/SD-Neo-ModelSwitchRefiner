/*
 * Model Chain -- LoRA Details: the dialog's clipboard and prompt helpers, the
 * trigger-word chips, the prompt cards' actions and the preset watcher.
 * Carried in from the user's SD-ForgeNeo-ExtendedLoraDetails extension;
 * docs/26-lora-details.md is the record.
 *
 * Everything here degrades quietly: if the host page changes shape, the buttons
 * simply do nothing rather than throwing inside a Gradio event handler.
 */

function mcLoraPromptTextarea(tabname, negative) {
    const suffix = negative ? "_neg_prompt" : "_prompt";
    const root = typeof gradioApp === "function" ? gradioApp() : document;
    if (!root) return null;
    return root.querySelector("#" + tabname + suffix + " > label > textarea");
}

function mcLoraSeparator() {
    try {
        if (typeof opts !== "undefined" && opts && opts.extra_networks_add_text_separator) {
            return opts.extra_networks_add_text_separator;
        }
    } catch (e) {
        /* opts is not published yet */
    }
    return ", ";
}

function mcLoraWriteToPrompt(tabname, text, mode, negative) {
    const textarea = mcLoraPromptTextarea(tabname, negative);
    if (!textarea) return;

    const value = (text || "").trim();
    if (!value) return;

    if (mode === "replace") {
        textarea.value = value;
    } else {
        const existing = textarea.value.trim();
        textarea.value = existing ? existing + mcLoraSeparator() + value : value;
    }

    if (typeof updateInput === "function") {
        updateInput(textarea);
    } else {
        textarea.dispatchEvent(new Event("input", { bubbles: true }));
    }
}

/* Append the selected prompt to the tab's positive prompt box. */
function mcLoraAppendPrompt(tabname, text) {
    mcLoraWriteToPrompt(tabname, text, "append", false);
    return [];
}

/* Overwrite the tab's positive prompt with the selected prompt. */
function mcLoraReplacePrompt(tabname, text) {
    mcLoraWriteToPrompt(tabname, text, "replace", false);
    if (typeof closePopup === "function") closePopup();
    return [];
}

/*
 * The panel's own buttons: copy, the trigger-word chips, the prompt cards.
 *
 * The panel is HTML that Gradio replaces every time the dialog opens, so one
 * listener on the document handles every [data-mc-lora-action] inside it. Each
 * section keeps the exact text its buttons act on in a hidden .mc-lora-copy-source.
 */

function mcLoraCopyFallback(value) {
    const area = document.createElement("textarea");
    area.value = value;
    area.setAttribute("readonly", "");
    area.style.position = "fixed";
    area.style.left = "-9999px";
    area.style.opacity = "0";
    document.body.appendChild(area);
    area.select();
    let copied = false;
    try {
        copied = document.execCommand("copy");
    } catch (e) {
        copied = false;
    }
    area.remove();
    return copied;
}

/* The clipboard API needs a secure context; a LAN address over http is not one. */
function mcLoraCopyText(text) {
    const value = String(text || "");
    if (navigator.clipboard && window.isSecureContext) {
        return navigator.clipboard.writeText(value).then(
            function () {
                return true;
            },
            function () {
                return mcLoraCopyFallback(value);
            }
        );
    }
    return Promise.resolve(mcLoraCopyFallback(value));
}

function mcLoraSourceText(scope) {
    const source = scope ? scope.querySelector(".mc-lora-copy-source") : null;
    if (!source) return "";
    return source.tagName === "TEXTAREA" ? source.value : source.textContent;
}

function mcLoraFlash(button, label) {
    if (!button || button.dataset.mcLoraLabel) return;
    button.dataset.mcLoraLabel = button.textContent;
    button.textContent = label;
    setTimeout(function () {
        button.textContent = button.dataset.mcLoraLabel;
        delete button.dataset.mcLoraLabel;
    }, 1200);
}

function mcLoraSplitWords(text) {
    return String(text || "")
        .split(/ *, */)
        .filter(function (word) {
            return word.trim();
        });
}

function mcLoraActivationField(section) {
    const id = section ? section.getAttribute("data-mc-lora-target") : "";
    if (!id) return null;
    const holder = mcLoraGradioRoot().querySelector("#" + CSS.escape(id));
    return holder ? holder.querySelector("textarea, input") : null;
}

function mcLoraSetField(field, value) {
    field.value = value;
    if (typeof updateInput === "function") {
        updateInput(field);
    } else {
        field.dispatchEvent(new Event("input", { bubbles: true }));
    }
}

/* Light up the chips whose word is in the activation text right now. */
function mcLoraSyncChips(section) {
    const field = mcLoraActivationField(section);
    if (!field) return;
    const words = mcLoraSplitWords(field.value);
    section.querySelectorAll(".mc-lora-word").forEach(function (chip) {
        chip.classList.toggle("mc-lora-word-on", words.indexOf(chip.getAttribute("data-word")) !== -1);
    });
}

function mcLoraTriggerAction(section, action, chip) {
    const field = mcLoraActivationField(section);
    if (!field) return;
    let words = mcLoraSplitWords(field.value);

    if (action === "toggle-trigger") {
        const word = chip.getAttribute("data-word");
        if (words.indexOf(word) !== -1) {
            words = words.filter(function (piece) {
                return piece !== word;
            });
        } else {
            words.push(word);
        }
        mcLoraSetField(field, words.join(", "));
    } else if (action === "use-triggers") {
        mcLoraSetField(field, mcLoraSourceText(section));
    } else if (action === "append-triggers") {
        mcLoraSplitWords(mcLoraSourceText(section)).forEach(function (word) {
            if (words.indexOf(word) === -1) words.push(word);
        });
        mcLoraSetField(field, words.join(", "));
    }
    mcLoraSyncChips(section);
}

/* Hand a card's request to Python through the hidden textbox + button pair. */
function mcLoraBridge(id, value) {
    const root = mcLoraGradioRoot();
    const area = root.querySelector("#" + CSS.escape(id) + " textarea");
    const button = root.querySelector("#" + CSS.escape(id + "_button"));
    if (!area || !button) return false;
    mcLoraSetField(area, value);
    button.click();
    return true;
}

function mcLoraCardAction(card, action, button) {
    const grid = card.closest("[data-mc-lora-bridge]");
    const tab = grid ? grid.getAttribute("data-mc-lora-tab") : "txt2img";
    const index = card.getAttribute("data-index");

    if (action === "append-prompt") {
        mcLoraAppendPrompt(tab, mcLoraSourceText(card));
        mcLoraFlash(button, "Appended");
    } else if (action === "replace-prompt") {
        mcLoraReplacePrompt(tab, mcLoraSourceText(card));
    } else if (action === "view" || action === "download") {
        if (!grid || !mcLoraBridge(grid.getAttribute("data-mc-lora-bridge"), action + ":" + index)) return;
        if (action === "download") {
            button.disabled = true;
            button.textContent = "Downloading…";
        } else {
            const panel = card.closest(".edit-user-metadata");
            const viewer = panel ? panel.querySelector(".mc-lora-picker") : null;
            if (viewer) viewer.scrollIntoView({ behavior: "smooth", block: "nearest" });
        }
    }
}

function mcLoraHandleAction(button) {
    const action = button.getAttribute("data-mc-lora-action");
    const card = button.closest(".mc-lora-card");
    if (card && action !== "copy") {
        mcLoraCardAction(card, action, button);
        return;
    }

    const section = button.closest(".mc-lora-section") || card;
    if (action === "copy") {
        mcLoraCopyText(mcLoraSourceText(card || section)).then(function (copied) {
            mcLoraFlash(button, copied ? "Copied" : "Copy failed");
        });
    } else if (action === "copy-word") {
        mcLoraCopyText(button.getAttribute("data-word")).then(function (copied) {
            mcLoraFlash(button, copied ? "Copied" : "Copy failed");
        });
    } else if (action === "toggle-trigger" || action === "use-triggers" || action === "append-triggers") {
        mcLoraTriggerAction(section, action, button);
    } else if (action === "append-negative") {
        const holder = button.closest("[data-mc-lora-tab]");
        mcLoraWriteToPrompt(holder ? holder.getAttribute("data-mc-lora-tab") : "txt2img", mcLoraSourceText(section), "append", true);
        mcLoraFlash(button, "Appended");
    } else if (action === "expand") {
        const open = section.classList.toggle("mc-lora-expanded");
        button.textContent = open ? "Show less" : "Show more";
        button.setAttribute("aria-expanded", open ? "true" : "false");
    }
}

/* Typing in the activation text keeps the chips' highlighting honest. */
document.addEventListener("input", function (event) {
    const field = event.target;
    const holder = field && field.closest ? field.closest("[id]") : null;
    if (!holder || !holder.id) return;
    mcLoraGradioRoot()
        .querySelectorAll('.mc-lora-triggers[data-mc-lora-target="' + CSS.escape(holder.id) + '"]')
        .forEach(mcLoraSyncChips);
});

document.addEventListener("click", function (event) {
    const target = event.target;
    const button = target && target.closest ? target.closest("[data-mc-lora-action]") : null;
    if (!button || !button.closest(".edit-user-metadata")) return;
    event.preventDefault();
    try {
        mcLoraHandleAction(button);
    } catch (e) {
        console.error("[Model Chain LoRA details]", e);
    }
});

/* Python clamps a text by its length; one that turns out to fit on screen needs
   neither the fade nor the "Show more". Measured once it is visible. */
function mcLoraFitClamps() {
    mcLoraGradioRoot()
        .querySelectorAll(".edit-user-metadata .mc-lora-clamp:not([data-mc-lora-fit])")
        .forEach(function (text) {
            if (!text.offsetParent || !text.clientHeight || text.closest(".mc-lora-expanded")) return;
            text.setAttribute("data-mc-lora-fit", "1");
            if (text.scrollHeight > text.clientHeight + 4) return;
            text.classList.remove("mc-lora-clamp");
            const toggle = text.parentElement ? text.parentElement.querySelector('[data-mc-lora-action="expand"]') : null;
            if (toggle) toggle.remove();
        });
}

if (typeof onAfterUiUpdate === "function") {
    onAfterUiUpdate(mcLoraFitClamps);
}

/*
 * UI Preset → network browsers.
 *
 * Forge renders each extra-network pane server-side, so switching the UI Preset
 * leaves the cards, the tree and the folder buttons showing the previous
 * preset's folders until the pane is rebuilt. The hidden per-tab refresh button
 * is the only path to that rebuild, so it is clicked here — the same thing the
 * pane's own ↻ control does.
 */

const mcLoraNetworkTabs = ["txt2img", "img2img"];
let mcLoraPresetLast = null;
let mcLoraPresetTimer = null;

function mcLoraGradioRoot() {
    return typeof gradioApp === "function" ? gradioApp() : document;
}

function mcLoraOption(name, fallback) {
    try {
        if (typeof opts !== "undefined" && opts && name in opts) return opts[name];
    } catch (e) {
        /* options are not published yet */
    }
    return fallback;
}

function mcLoraPresetValue() {
    const root = mcLoraGradioRoot();
    if (!root) return null;
    const holder = root.querySelector("#forge_ui_preset");
    if (!holder) return null;
    const field = holder.querySelector("input, select, textarea");
    if (!field) return null;
    return (field.value || "").trim();
}

/* Rebuild every network pane. Safe to call when the panes do not exist yet. */
function mcLoraRefreshExtraNetworks() {
    const root = mcLoraGradioRoot();
    if (!root) return [];

    mcLoraNetworkTabs.forEach(function (tabname) {
        // One click rebuilds every page of that tab, so the first button wins.
        const button = root.querySelector('[id^="' + tabname + '_"][id$="_extra_refresh_internal"]');
        if (button) button.dispatchEvent(new Event("click"));
    });
    return [];
}

function mcLoraWatchPreset() {
    if (mcLoraOption("model_chain_lora_folder_filter_enabled", true) === false) return;
    if (mcLoraOption("model_chain_lora_folder_filter_auto_refresh", true) === false) return;

    const value = mcLoraPresetValue();
    if (value === null || value === "") return;

    if (mcLoraPresetLast === null) {
        mcLoraPresetLast = value; // first sighting: the panes already match it
        return;
    }
    if (value === mcLoraPresetLast) return;

    mcLoraPresetLast = value;
    // The dropdown's text changes while it is being typed in, so settle first
    // and rebuild once.
    clearTimeout(mcLoraPresetTimer);
    mcLoraPresetTimer = setTimeout(mcLoraRefreshExtraNetworks, 600);
}

if (typeof onUiLoaded === "function") {
    onUiLoaded(function () {
        mcLoraPresetLast = mcLoraPresetValue();
    });
}

if (typeof onAfterUiUpdate === "function") {
    onAfterUiUpdate(mcLoraWatchPreset);
}
