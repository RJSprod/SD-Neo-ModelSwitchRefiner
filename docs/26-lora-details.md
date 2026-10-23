# LoRA Details — the standalone extension carried in, and the dialog the width of the window

The user's `SD-ForgeNeo-ExtendedLoraDetails` extension — a CSV library of what
Civitai says about each LoRA, the fetch that fills it, the gallery images
behind the prompts, the folders each UI Preset shows, and a panel of all of it
inside Forge's LoRA dialog — lives here now, as `mc_lora_*` and
`mc_extra_networks_*`. This file is the design: what was asked, what was
decided, how the dialog is laid out, where the files went, and what is
still open. Behaviour people see is in `README.md` ("LoRA Details"); the
reasoning behind a line of code is in that module's docstring.

The brief was the user's, 2026-10-23, after a screenshot of the standalone's
new panel: "it still renders everything in a narrow column in the middle of
the page." Three asks, one pull request, no phases:

1. "Move all the functionality of lora extender into the modelswitchrefiner so
   I can uninstall the lora extender repo. Make sure we keep the nice
   organization of the modelswitchrefiner repo."
2. "Update the view so that extra networks is a full page (full width) view
   instead of the current narrow strip. Feel free to completely redesign ...
   compact and easy to review content ... group similar items, as well as
   handle when content is missing. Do not remove any existing features."
3. "'show / hide' folders doesn't work in the extra networks drawer, so lets
   make sure that works. ... I'm okay with hiding all subfolders in general
   from the folder buttons. I made all lora folders model names, like SDXL,
   and I do have some subfolders like SDXL\LIGHTING, but I have no need to tap
   on it directly."

---

## 1. Decisions

| Question | Decision |
|---|---|
| Where the standalone's modules go | Flat `mc_lora_*.py` and `mc_extra_networks_*.py` at the root, one module per concern, like every other area; no package. `mc_lora_common.py` holds the logger, the settings reader and the normalisers the others share |
| The log | Through the `model_chain` logger, every line opening `Model Chain LoRA details:`; never a path, a file's name or a user's text in a line (the standing convention since handoff 47). Every line the standalone wrote with a path was rewritten to say how many, or what kind |
| Where the files go | One data root, `<WebUI data>/model_chain_lora/`, like the LLM's and the Voice Chat's: `library/`, `images/`, `cache/` (hashes, thumbnails), `preset_folders.json`, `last_used.json`. The library and the image folder can each be pointed elsewhere by a setting |
| The settings | Three sections — **Model Chain: LoRA Details**, **Model Chain: Civitai fetch**, **Model Chain: Extra networks** — every key the standalone's with `eld_` read `model_chain_lora_`, same defaults; a sentence or two under each, as the settings test holds the other sections to. Three of the standalone's settings have no counterpart: the panel they described (opened by default, opened on a match, drawn above the fields) no longer exists |
| The carry-over | Once, on the first start with the settings loaded: the standalone's files copied into the data root where nothing of the name is there yet, a setting still at its default here taking the standalone's value, the record in a hidden option. Nothing deleted; the user uninstalls the standalone. While it is still installed a log line says so, because both would draw into the dialog |
| The dialog | Not a panel under Forge's fields: the whole dialog laid out again, Forge's components moved into the first of three columns under one head row, the width of the window with the theme and without it. Nothing of Forge's rebuilt, so Save saves as before |
| The folder row | A stylesheet fix at the theme's own specificity, and a setting, on by default, that keeps the folder buttons to the top level |
| Tests | Every test of the standalone's ported into pytest style under the `host` and `lora_root` fixtures, and new ones for what was built: the move, the carry-over, the settings sections, the stylesheet and the script in Chromium |

---

## 2. The map

| Standalone (`extended_lora_details/`) | Here | Holds |
|---|---|---|
| `common.py` | `mc_lora_common.py`, `mc_lora_paths.py` | the logger, `opt()`, the host's LoRA folders, normalisers; the data root and every path under it |
| `hashing.py` | `mc_lora_hashing.py` | SHA256 and AddNet hashes, cached in `cache/hashes.json`, resolved when first needed |
| `library.py` | `mc_lora_library.py` | the CSV rows, the aliases a column may go by, matching by hash, short hash, path, name, stem |
| `civitai.py` | `mc_lora_civitai.py` | the client, the base-model gate, MOSS-Image by tensor names, the scan of a folder into a CSV |
| `descriptions.py` | `mc_lora_descriptions.py` | a model page's HTML as text (its extractor's state out of `HTMLParser`'s way) |
| `layouts.py` | `mc_lora_layouts.py` | MOSS-Image layout prompts, parsed and labelled, kept verbatim |
| `images.py` | `mc_lora_images.py` | downloads into one flat folder, presence read from disk, thumbnails under `cache/thumbs` |
| `jobs.py` | `mc_lora_jobs.py` | the one-at-a-time job runner: a fetch, a hash run, image downloads |
| `state.py` | `mc_lora_last_used.py` | the key and folder of the last fetch, owner-only |
| `render.py` | `mc_lora_render.py` | every HTML fragment, classes `mc-lora-*` |
| `details_ui.py` | `mc_lora_details.py` | the dialog: gathering, the move, population, the fetch button, the cards' bridge |
| `settings_ui.py`, `scripts/extended_lora_details.py` | `mc_lora_settings.py` | the three sections, the uploader widget, the carry-over |
| `tab_ui.py` | `mc_lora_tab.py` | the **LoRA Details** tab: Library, Preset folders, Images, Fetch from Civitai, Jobs |
| `preset_folders.py` | `mc_extra_networks_folders.py` | folder assignments per UI Preset and network page |
| `folders_ui.py` | `mc_extra_networks_folders_ui.py` | the editor of those assignments, on the tab |
| `network_filter.py` | `mc_extra_networks_filter.py` | the pages narrowed to the assignments; the folder buttons: assigned, and top-level only |
| `javascript/extended-lora-details.js` | `javascript/model_chain_lora_details.js` | the chips, the clipboard, the cards' actions, the preset watcher |
| `style.css` | `style.css`, section "Model Chain -- LoRA Details" | the fragments' rules, and the dialog's |
| `tools/civitai_lora_csv.py` | `tools/civitai_lora_csv.py` | the fetch from a terminal |

`scripts/model_chain.py` registers the three sections and the hidden record,
wraps the page class and the editor class before the page is built
(`on_before_ui`), announces the tab (`on_ui_tabs`), runs the carry-over and
loads the library once the settings are up (`on_app_started`), and cancels
the jobs and writes the hash cache out when the script is unloaded.

---

## 3. The dialog

### 3.1 Why the panel could not be widened

Forge builds a LoRA's *edit metadata* popup in `LoraUserMetadataEditor.create_editor()`:
one `Group` with class `edit-user-metadata`, a row of the name, the
description, the file table and the preview, then the training tags, the
activation text, the weight, the negative prompt, the random prompt and the
notes, then `create_default_buttons()` — Cancel, Replace preview, Save.
Forge's stylesheet gives the box 56 em. The theme (docs/24) draws the popup as
a dialog: `.global-popup` padded 24 px, its inner wrapper scrolling, and the
box `min(56em, calc(100vw - 48px))` wide, at specificity (0,3,0). The
standalone's `.edit-user-metadata:has(.eld-panel) { width: min(84em, 94vw) }`
is (0,2,0): under the theme it never applied, and the panel under the notes
stayed in the 56 em column. Hence the screenshot.

### 3.2 The move

`create_default_buttons` is the one moment Forge gives an extension inside
the dialog with every stock component built and the box still open, so
`mc_lora_details.install()` wraps it (once; `Reload UI` re-runs every script
and leaves `ui_edit_user_metadata` in `sys.modules`), and `build_dialog()`
lays the whole thing out again:

- a head row (`mc-lora-head-row`): Forge's name for the card first, then the
  title strip — Civitai's name, version, base model and creator as pills, the
  link to the page, and how the row was matched;
- a body row (`mc-lora-body`) of three columns, 3 : 4 : 6, each a card as tall
  as what it holds:
  - **File** (`mc-lora-col-file`, 280 px at least): Forge's preview and file
    table, then every field you edit in Forge's order — description, preset,
    training tags, activation text, weight, negative prompt, notes — then
    whatever else Forge or another extension put in the box before this ran
    (the random prompt row among them), in the order it came;
  - **Civitai** (`mc-lora-col-about`, 320 px): the facts, the fetch button
    when there is work for it, and the side HTML: trigger-word chips, the
    description and version notes, the negative prompt, every field of the
    row and the raw row folded away;
  - **Gallery prompts** (`mc-lora-col-gallery`, 480 px, hidden without
    prompts): the selector with stepping, the selected prompt beside its
    image with its downloads, Append and Replace, and every prompt as a
    card;
- Forge's own buttons last, across the whole width, because Forge makes them
  after the wrapper returns.

Gradio keeps the page's layout as a tree of `children` lists, read when the
configuration is built (`Blocks.get_config_file`, at launch); a component's
`parent` is the context it was made in. Moving one is taking it out of one
list and appending it to another with `parent` to match — the two writes
Gradio's own `with` blocks make — so every component keeps its id, its events
and its place in Forge's `put_values_into_components` and save handlers. A
text box Gradio has already wrapped in a pseudo `Form` (`fill_expected_parents`,
at the row's close) is moved out of that form, and a form left empty is
dropped with it; the moved boxes get their forms back by the same call on the
column. The hidden name box and the hidden button Forge opens the dialog with
stay where they are, as does any block that holds nothing visible. The row
Forge drew, now empty, is detached. The box gets the class `mc-lora-dialog`,
which is what the stylesheet widens under.

Should any of it fail — `edit_name` not found, the row not where Forge drew
it, an exception — the dialog is left exactly as Forge built it, with our
rows appended under the notes as the standalone's panel was, and the console
says so. Population is a *second* listener on the same `button_edit` click
Forge uses, so Forge's outputs are untouched and the stock dialog works if
ours raises. The whole wrapper is behind one setting, **Draw a LoRA card's
details dialog the width of the window**, on by default.

### 3.3 The width

In the stylesheet's section: the popup's inner wrapper, Forge's inline-block
at most 90% wide with 2 em of padding, is lifted to the whole width with 24 px
of padding by a `:has()` rule while it holds the laid-out box (the theme pads
the popup itself, so under it the wrapper's padding is 0); the box is 100% of
that, at specificity (0,4,0) under the theme, above the theme's (0,3,0). The
dialog is 1552 px wide in a 1600 px window either way; without the class —
the fallback, and the checkpoint and embedding editors — Forge's 56 em stands.
Gradio's Group wraps its children in a `.styler`, so no rule asks for a child
of the box; and Gradio gives every block `width: 100%`, which in a wrapping
row is a line of its own, so the name block is `width: auto` — the one rule
the first mutation check caught missing.

### 3.4 Verified

A Gradio 4.40 harness with Forge Neo's own editor, Forge's `popup()` and the
real `style.css`, in Chromium, with the theme on and off: the dialog the
window less 48 px, the three columns 356 / 467 / 687 px wide, the preview card
the column's width or gone when empty, and the standalone's 49 interaction
checks — chips, Copy, Use, Append, typing re-syncing the chips, Show more, the
negative prompt into its box, the raw row as JSON, the viewer's stepping and
the layout prompt's summary, the cards' Copy / Append / Replace / view /
download, the viewer's download and the heading's count, Replace closing the
dialog, the unmatched LoRA's fetch button queueing a job and reading *Retry*
once Civitai had nothing — all passing. Not run on the user's machine.

---

## 4. The folder row and the buttons

Forge's Tree View button (`extraNetworksControlTreeViewOnClick`) toggles
`extra-network-dirs-hidden` on the page's pane, and Forge's stylesheet hides
the folder row under it: `.extra-network-dirs-hidden .extra-network-dirs {
display: none }`, (0,2,0). The theme draws the row as a flex box of small
buttons with `:root[data-mc-theme="on"] .mc-theme-networks-page
.extra-network-dirs { display: flex }`, (0,4,0): the toggle flipped the class
and nothing changed. The fix is one rule beside the theme's at (0,5,0), hiding
the row when the pane carries the class. `tests/test_lora_css.py` toggles the
class in Chromium under the theme and without it.

The buttons: `create_dirs_view_html` walks every folder off disk and writes
one button per folder, nested ones as `SDXL/LIGHTING/`. The wrapper in
`mc_extra_networks_filter` now keeps the buttons whose label has one path
part — `SDXL/`, `/Flux/`, `Krea 2\` whichever way Forge spells it, and the
"all" button — before the preset filter runs. A LoRA in `SDXL/LIGHTING` is
still listed when `SDXL` is pressed, because Forge's button searches by path
prefix; what goes is the button the user never presses. **Show only top-level
folders in the folder buttons**, on by default, turns it off. The Tree style
of the browser draws no buttons and is untouched.

---

## 5. The carry-over

`mc_lora_settings.import_standalone(opts, extensions_dir, defaults=...)`,
from `on_app_started`, when the hidden record (`model_chain_lora_import_record`)
is empty and either the standalone's folder is under `extensions/` (found by
name, case-insensitively) or an `eld_*` key is in `config.json`:

- files, where nothing of the name is here yet, never over one that is:
  `library/` and `images/` tree by tree, `preset_folders.json`,
  `last_used.json` (made owner-only again, it holds the API key) and
  `.cache/hashes.json` into `cache/hashes.json`; the standalone's code never;
- settings, each `eld_*` to its `model_chain_lora_*`: written where ours is
  still at its default and the standalone's differs from it, kept as set
  where ours was set here, and `eld_panel_first`, `eld_open_by_default`,
  `eld_open_on_match` left behind;
- the record: version, the folder it came from, when, what was copied and
  kept by kind, which settings were applied (the API key's name never
  recorded) and which kept; `opts.save()` after.

A copy that fails writes no record, so the next start tries again. The log
says how many files and settings came over and that the extension can be
uninstalled; while it is still installed, every start says its own panel and
filter run beside these.

---

## 6. Tests

`tests/test_lora_library.py` (hashing, rows, matching, descriptions, layouts,
the fragments), `tests/test_lora_fetch.py` (base models, MOSS-Image, the scan,
the jobs, the remembered key and folder), `tests/test_lora_images.py` (the
downloads against a local server, thumbnails, the bulk job),
`tests/test_extra_networks_folders.py` (the rules, the filter, the buttons,
attaching to Forge), `tests/test_lora_dialog.py` (the move against a stand-in
of Gradio's block tree, population from a real library, the fetch plan, the
patch), `tests/test_lora_settings.py` (the sections, the carry-over, the
paths), `tests/test_lora_tab.py`, `tests/test_lora_css.py` and
`tests/test_lora_details_js.py` (the stylesheet and the script in Chromium).
The `lora_root` fixture in `tests/conftest.py` puts the data root under
`tmp_path` and forgets the two stores that cache their path; the checkout
guard at the bottom of that file now watches `model_chain_lora/` too. Nine
mutations were tried; eight failed a test at once and the ninth (the name
block's `width: auto`) got its assertion.

---

## 7. Open

- The standalone extension, while still installed, wraps the same two Forge
  classes; the dialog then carries its accordion under the notes as well. The
  log says so at every start. Uninstalling it is the user's step.
- `:has()` carries the wrapper's width; a browser without it (none the user
  runs) gets the theme's dialog width with the three columns inside it.
- The gallery column's cards grid is `auto-fill` at 13.5 rem; the Gradio
  `Image` of the selected prompt is capped at 24 em. Neither was tuned on the
  user's screen.
