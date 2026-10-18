// Model Chain -- Gallery Quadrant.
//
// The user's own sd-gallery-quadrant extension, carried in as it was so that
// one extension is installed instead of two (mc_gallery_quadrant.py says why
// and what changed). It splits the gallery preview image into five
// interaction zones:
//
//   ┌───────────┬───────────┬───────────┐
//   │⬇ Download │  ✕ Unfocus│  💾 Save  │
//   ├─────────────────┬─────────────────┤
//   │     ◀ Prev      │      Next ▶     │
//   └─────────────────┴─────────────────┘
//
// Enable/disable via Settings → Model Chain: Gallery Quadrant. The one
// change from the standalone: the top-left zone's destination is a setting
// there -- the browser's downloads, as before, or a folder on this machine
// that the WebUI writes the picture into over a route of this extension's
// (ROUTE below); the zone then says "⬇ Folder" and shows what happened.
//
// The class names and the data markers are the standalone's: a page where it
// is still enabled draws one overlay, because whichever script reaches a
// gallery first marks it and the other skips it.

(function () {
    "use strict";

    var HINT_DURATION_MS      = 500;
    var HIGHLIGHT_DURATION_MS = 250;
    var REFUSED_DURATION_MS   = 600;

    // The options Python registers (mc_gallery_quadrant.py).
    var OPT_ENABLED     = "model_chain_gallery_quadrant";
    var OPT_DESTINATION = "model_chain_gallery_quadrant_destination";
    var DESTINATION_FOLDER = "A folder on this machine";
    var ROUTE = "/model-chain/gallery-quadrant/save";

    var ZONE_TL = 0;
    var ZONE_TM = 1;
    var ZONE_TR = 2;
    var ZONE_BL = 3;
    var ZONE_BR = 4;

    var TITLE_DOWNLOAD = "⬇ Download";
    var TITLE_FOLDER   = "⬇ Folder";

    function setting(name, fallback) {
        try {
            if (typeof opts === "undefined" || opts === null) return fallback;
            var value = opts[name];
            return value === undefined || value === null ? fallback : value;
        } catch (error) {
            return fallback;
        }
    }

    function enabled() {
        return !!setting(OPT_ENABLED, true) && !setting("js_modal_lightbox", false);
    }

    function savesToFolder() {
        return setting(OPT_DESTINATION, "") === DESTINATION_FOLDER;
    }

    // ------------------------------------------------------------------ //
    //  Browser download (top-left, the browser's downloads)
    // ------------------------------------------------------------------ //
    function galleryDownloadToBrowser(imgSrc) {
        fetch(imgSrc)
            .then(function (r) { if (!r.ok) throw new Error("HTTP " + r.status); return r.blob(); })
            .then(function (blob) {
                var url = URL.createObjectURL(blob);
                var a = document.createElement("a");
                a.href = url;
                a.download = imgSrc.split("?")[0].split("/").pop() || "image.png";
                a.style.display = "none";
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
                setTimeout(function () { URL.revokeObjectURL(url); }, 1000);
            })
            .catch(function (e) { console.error("[GQA] download failed:", e); });
    }

    // ------------------------------------------------------------------ //
    //  Save to the folder (top-left, a folder on this machine)
    //  The page posts the preview's src -- a Gradio file URL -- and the server
    //  finds the file it names under Forge's output folders and copies it into
    //  the folder the settings name. The tap shows nothing but the zone's own
    //  flash, as Forge's Save in the top-right zone does ("I don't want there
    //  to be any dialogue when I tap, I just want the transparent tap
    //  indication"); a refusal tints the zone red for a moment and its reason
    //  goes to the console.
    // ------------------------------------------------------------------ //
    function gallerySaveToFolder(imgSrc, galleryEl) {
        var cell = galleryEl.querySelector(".gqa-overlay .gqa-tl");
        return fetch(ROUTE, {
            method: "POST",
            credentials: "same-origin",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({src: imgSrc}),
        }).then(function (r) {
            return r.json().catch(function () { return {}; }).then(function (body) {
                if (r.ok && body && body.ok) return;
                var reason = (body && body.error) || ("HTTP " + r.status);
                console.error("[GQA] save to folder refused:", reason);
                showRefused(cell);
            });
        }).catch(function (e) {
            console.error("[GQA] save to folder failed:", e);
            showRefused(cell);
        });
    }

    function showRefused(cell) {
        if (!cell) return;
        cell.classList.add("gqa-refused-visible");
        if (cell.gqaRefusedTimer) clearTimeout(cell.gqaRefusedTimer);
        cell.gqaRefusedTimer = setTimeout(function () {
            cell.gqaRefusedTimer = null;
            cell.classList.remove("gqa-refused-visible");
        }, REFUSED_DURATION_MS);
    }

    // ------------------------------------------------------------------ //
    //  Exit preview (top-middle)
    //  Gradio 4.40 Gallery.svelte: handle_click(i) sets selected = selected===i ? null : i
    //  So clicking the already-selected thumbnail button toggles it to null → grid view.
    //  selected_gallery_button() (defined in ui.js) returns that element.
    //  NOTE: do NOT dispatch any synthetic keyboard events here — global keydown
    //  listeners in edit-attention.js / edit-order.js call composedPath() on synthetic
    //  events and crash with "target.matches is not a function".
    // ------------------------------------------------------------------ //
    function galleryExitPreview(galleryEl) {
        // Gradio 4.40 renders a button[aria-label="Clear"] inside the gallery
        // when in preview mode. Clicking it returns to the grid view.
        var clearBtn = galleryEl.querySelector("button[aria-label='Clear']");
        if (clearBtn) {
            clearBtn.click();
        }
    }

    // ------------------------------------------------------------------ //
    //  Zone index
    // ------------------------------------------------------------------ //
    function zoneAt(relX, relY, width, height) {
        var isTop = relY < height / 2;
        if (isTop) {
            var third = width / 3;
            if (relX < third)     return ZONE_TL;
            if (relX < third * 2) return ZONE_TM;
            return ZONE_TR;
        }
        return relX < width / 2 ? ZONE_BL : ZONE_BR;
    }

    function getZoneIndex(evt, imageButton) {
        // Use the image-button rect (the actual image area, excluding icon-buttons overlay)
        var rect = imageButton.getBoundingClientRect();
        return zoneAt(evt.clientX - rect.left, evt.clientY - rect.top, rect.width, rect.height);
    }

    // ------------------------------------------------------------------ //
    //  Action dispatcher
    // ------------------------------------------------------------------ //
    function galleryZoneAction(evt, imageButton, galleryEl) {
        var zone = getZoneIndex(evt, imageButton);
        if (zone === ZONE_TL) {
            var img = imageButton.querySelector("img");
            if (!img) return;
            if (savesToFolder()) gallerySaveToFolder(img.src, galleryEl);
            else galleryDownloadToBrowser(img.src);
        } else if (zone === ZONE_TM) {
            galleryExitPreview(galleryEl);
        } else if (zone === ZONE_TR) {
            saveImage();
        } else if (zone === ZONE_BL) {
            modalImageSwitch(-1);
        } else {
            modalImageSwitch(1);
        }
    }

    // ------------------------------------------------------------------ //
    //  Per-gallery delegated click listener
    // ------------------------------------------------------------------ //
    function findImageButton(target, galleryEl) {
        // Returns the BUTTON.image-button inside button.preview if the click
        // landed on the image area. Returns null if the click was on:
        //   - a thumbnail (inside .thumbnails or .grid-wrap)
        //   - the icon-buttons overlay (Gradio's own Download/Clear icons)
        //   - anything outside button.preview
        var iconButtons = galleryEl.querySelector(".icon-buttons");
        if (iconButtons && iconButtons.contains(target)) return null; // let Gradio handle

        var thumbnails = galleryEl.querySelector(".thumbnails");
        if (thumbnails && thumbnails.contains(target)) return null;

        var gridWrap = galleryEl.querySelector(".grid-wrap");
        if (gridWrap && gridWrap.contains(target)) return null;

        // Walk up to find BUTTON.image-button inside button.preview
        var el = target;
        while (el && el !== galleryEl) {
            if (el.tagName === "BUTTON" && el.classList.contains("image-button")) return el;
            el = el.parentElement;
        }
        return null;
    }

    function attachGalleryListener(galleryEl) {
        if (galleryEl.dataset.gqaAttached) return;
        galleryEl.dataset.gqaAttached = true;
        galleryEl.addEventListener("click", function (evt) {
            if (!enabled())       return;
            if (evt.button !== 0) return;
            var imageBtn = findImageButton(evt.target, galleryEl);
            if (!imageBtn) return;
            evt.preventDefault();
            evt.stopPropagation();
            evt.stopImmediatePropagation();
            galleryZoneAction(evt, imageBtn, galleryEl);
        }, true);
    }

    // ------------------------------------------------------------------ //
    //  Overlay: titles flash on gallery entry
    // ------------------------------------------------------------------ //
    function setupOverlayHints(previewButton, overlay) {
        var timer = null, showing = false, armed = true;
        previewButton.addEventListener("mouseenter", function () {
            if (!armed || showing) return;
            showing = true; armed = false;
            overlay.classList.add("gqa-titles-visible");
            timer = setTimeout(function () {
                overlay.classList.remove("gqa-titles-visible");
                showing = false; timer = null;
            }, HINT_DURATION_MS);
        });
        previewButton.addEventListener("mouseleave", function () {
            armed = true;
            if (timer) { clearTimeout(timer); timer = null; }
            overlay.classList.remove("gqa-titles-visible");
            showing = false;
        });
    }

    // ------------------------------------------------------------------ //
    //  Overlay: cell highlight flashes on zone entry (tracked via mousemove)
    // ------------------------------------------------------------------ //
    function setupCellHighlights(previewButton, cells) {
        var timers  = cells.map(function () { return null; });
        var showing = cells.map(function () { return false; });
        var armed   = cells.map(function () { return true; });
        var lastIdx = -1;

        function showCell(idx) {
            if (!armed[idx] || showing[idx]) return;
            showing[idx] = true; armed[idx] = false;
            cells[idx].classList.add("gqa-highlight-visible");
            timers[idx] = setTimeout(function () {
                cells[idx].classList.remove("gqa-highlight-visible");
                showing[idx] = false; timers[idx] = null;
            }, HIGHLIGHT_DURATION_MS);
        }
        function resetCell(idx) {
            armed[idx] = true;
            if (timers[idx]) { clearTimeout(timers[idx]); timers[idx] = null; }
            cells[idx].classList.remove("gqa-highlight-visible");
            showing[idx] = false;
        }

        previewButton.addEventListener("mousemove", function (evt) {
            // Use the image-button's rect so zone boundaries match click detection
            var imageBtn = previewButton.querySelector("button.image-button");
            var referenceEl = imageBtn || previewButton;
            var idx = getZoneIndex(evt, referenceEl);
            if (idx !== lastIdx) {
                if (lastIdx !== -1) resetCell(lastIdx);
                lastIdx = idx;
                showCell(idx);
            }
        });
        previewButton.addEventListener("mouseleave", function () {
            if (lastIdx !== -1) resetCell(lastIdx);
            lastIdx = -1;
        });
    }

    // ------------------------------------------------------------------ //
    //  Inject overlay DOM once per preview button
    // ------------------------------------------------------------------ //
    function attachQuadrantOverlay(previewButton) {
        if (previewButton.dataset.gqaOverlay) return;
        if (!enabled()) return;
        previewButton.dataset.gqaOverlay = true;

        var overlay = document.createElement("div");
        overlay.className = "gqa-overlay";
        var cells = [];
        [
            { cls: "gqa-tl", text: savesToFolder() ? TITLE_FOLDER : TITLE_DOWNLOAD },
            { cls: "gqa-tm", text: "✕ Unfocus"  },
            { cls: "gqa-tr", text: "💾 Save"     },
            { cls: "gqa-bl", text: "◀ Prev"      },
            { cls: "gqa-br", text: "Next ▶"      },
        ].forEach(function (z) {
            var cell = document.createElement("div");
            cell.className = "gqa-cell " + z.cls;
            cell.textContent = z.text;
            overlay.appendChild(cell);
            cells.push(cell);
        });
        previewButton.appendChild(overlay);
        setupOverlayHints(previewButton, overlay);
        setupCellHighlights(previewButton, cells);
    }

    // The top-left title follows the destination setting, which can change
    // without a reload. Written only when it differs, and not over a result
    // being shown: this runs after every update the host makes.
    function syncTitle(previewButton) {
        var cell = previewButton.querySelector(".gqa-overlay .gqa-tl");
        if (!cell || cell.gqaResultTimer) return;
        var wanted = savesToFolder() ? TITLE_FOLDER : TITLE_DOWNLOAD;
        if (cell.textContent !== wanted) cell.textContent = wanted;
    }

    // ------------------------------------------------------------------ //
    //  onAfterUiUpdate
    // ------------------------------------------------------------------ //
    function update() {
        ["txt2img_gallery", "img2img_gallery", "extras_gallery"].forEach(function (id) {
            var el = gradioApp().getElementById(id);
            if (el) attachGalleryListener(el);
        });
        gradioApp().querySelectorAll(".gradio-gallery > button.preview").forEach(function (previewButton) {
            attachQuadrantOverlay(previewButton);
            syncTitle(previewButton);
        });
    }

    if (typeof onAfterUiUpdate === "function") {
        onAfterUiUpdate(function () {
            try { update(); } catch (error) { console.error("[GQA] update failed:", error); }
        });
    }

    window.mcGalleryQuadrant = {
        OPT_ENABLED: OPT_ENABLED, OPT_DESTINATION: OPT_DESTINATION,
        DESTINATION_FOLDER: DESTINATION_FOLDER, ROUTE: ROUTE,
        TITLE_DOWNLOAD: TITLE_DOWNLOAD, TITLE_FOLDER: TITLE_FOLDER,
        zoneAt: zoneAt, update: update,
    };
})();
