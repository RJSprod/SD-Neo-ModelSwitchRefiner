# Hosting the Lobe theme — a review of `sd-webui-lobe-theme` 3.5.4, and what it takes

Written 2026-10-13 against the user's own copy of `lobehub/sd-webui-lobe-theme`
(the `main` branch as a zip: `package.json` says 3.5.4, the changelog's last
entry is 2024-05-24) and against Forge Neo's loader as read the same day from
`Haoming02/sd-webui-forge-classic`, branch `neo` (`modules/ui_gradio_extensions.py`,
`modules/scripts.py`, `modules/extensions.py`).

The question asked: *could this extension be the theme's host — run it, import
everything Lobe has including my settings, so I can uninstall the original and
only this extension controls the theme — so seamlessly that Mini Paint NEO
needs no change?*

The short answer is **yes**, and most of it is less work than it looks, because
Forge does not know which extension a theme is: it loads every enabled
extension's `javascript/*.js` and `style.css` and nothing else distinguishes
the Lobe folder. The bundle is the theme; whoever serves its five routes and
keeps its one settings file is its host. The two things that need care are the
hand-over itself (one settings file to import, and one load where both copies
could mount) and the licence. Section 7 has the plan; section 8 what to decide.

Nothing in this document has run on the user's machine. The first goal the user
set — fewer bright spots at night, Generate first — was built in the same
session as this review, as this extension's own stylesheet and a setting
(`README.md`, *A quieter night mode*; `tests/test_night_mode.py`), and does
not wait on anything below.

---

## 0. The decision, and what was built instead (2026-10-13, same day)

The user read section 8 and declined the licence: *"i really dont want that
license. Can we just recreate the parts of the theme i want, reverse
engineered?"* So **sections 4 and 7 below are retired** — nothing of the
Lobe extension is vendored, served or patched — and what was built is an
independent theme of this extension's own, from a description of the page and
from Forge's own DOM, with nothing taken from `src/`. The user then sent their
`lobe_theme_config.json`, which is the specification of what they had on and
is read once, as their data, to start the new switches where theirs were.

What that file says, and what became of each line:

| Lobe setting | Value | Here |
|---|---|---|
| `layoutSplitPreview` | true | **Generate above the gallery, prompts at the top of the settings column** (`model_chain_theme_split`) |
| `enableSidebar`, `sidebarExpand`, `sidebarWidth` | true, false, 280 | **Quick settings** in a drawer, collapsed, 280 px (`model_chain_theme_quick`, `_quick_width`) |
| `promptTextareaType` | resizable | **Resizable prompt boxes** (`model_chain_theme_prompts`) |
| `svgIcon` | true | **Drawn icons** on the tool buttons (`model_chain_theme_icons`), buttons whose whole text is the emoji and nothing else |
| `layoutHideFooter` | true | **Hide the WebUI footer**, which this extension already had |
| `confirmPageUnload` | false | **Ask before leaving the page** (`model_chain_theme_confirm_unload`) |
| `primaryColor`, `neutralColor` | none | the automatic accent: white at night, black by day (`model_chain_theme_accent`) |
| `logoCustomTitle` | none | **Title in the header** (`model_chain_theme_title`, "Forge") |
| `enableExtraNetworkSidebar`, `extraNetworkSidebarExpand`, `extraNetworkSidebarWidth`, `extraNetworkCardSize` | true, false, 340, 86 | **The extra-network pages in a drawer** on the right, collapsed, 340 px, 86 px cards (`model_chain_theme_networks`, `_networks_width`, `_networks_card`). Built one round after the first import: a record from that round is read again for these (`IMPORT_VERSION`) |
| `enableImageInfo` | true | **Generation info as a panel** (`model_chain_theme_image_info`): the prompt and the negative prompt in blocks with copy buttons, every setting a chip, Forge's paragraph kept out of view for what reads it |
| `enableWebFont` | true | a **Typeface** setting of ours (`model_chain_theme_font`, a Google Fonts family; empty is the system's) |
| `extraNetworkFixedMode` | fixed | **The extra networks as a column** beside the generation UI (`model_chain_theme_networks_mode`: *column*; *drawer* is Lobe's float). Built two rounds after the first import, read from the record's `raw` by the third version |
| `i18n`, `liteAnimation`, `promptEditor`, `enableHighlight`, `logoType`, `logoCustomUrl`, `sidebarFixedMode` | — | not built: translations, the background glow, Lobe's prompt editor and highlighter, its logo, the quick settings as a fixed column |
| `detail` | "Not Found" | a 404 body the theme once saved as a setting; ignored |

**Applied quietly (2026-10-14).** `apply` runs on every after-update, and
Forge raises the next after-update 250 ms after any change to a child list
inside its app element: the first build wrote the title's text and rebuilt
the ⋯ menu on every apply, so the page updated every 300 ms on every tab for
as long as it was open -- a 50 ms stall under every drag of the Forge
Assistant, in the user's own log. Every write is conditional now (`show`,
`setClass`, `menuFor`), and `tests/test_theme_layout.py` holds that an apply
to a settled page changes no child list. docs/22 §3.36 has the measurements.

**The extra-network pages, in place (2026-10-14).** The user's second run of
the drawer: the live search no longer filtered as they typed, the search box
sat loose, Lobe's fixed-column layout was missing, and the drawer stayed open
on tabs that have no pages. Forge Neo's `extraNetworks.js` filters a page's
cards with `#<tab>_extra_tabs div.card` and shows a page's controls with
`#<tab>_extra_tabs .extra-networks-controls-div > div`: the first build had
moved the pages into an aside outside that element, so the search found no
cards. The pages stay Forge's own tab panels now, the drawer's head (title,
close, page strip, drag edge, and Forge's controls strip moved only within)
is one more child of `#<tab>_extra_tabs`, and the stylesheet draws the shown
page under the head -- fixed over the page's right edge, or sticky in the
second column of a grid the host becomes (`data-mc-theme-networks="column"`).
One drawer per generation tab, which goes with its tab; the edge drags to a
width remembered per browser until the width setting changes; `tests/
test_theme_layout.py` runs Forge's own filter over the drawer's search box.

**The page's margins (2026-10-14).** From three of the user's screens: the
content 8 px from the display's edge under the Forge Assistant's focus mode
(its workspace root pads 8 px) and about four times that outside it, the tab
bar inset with it. Gradio 4.40 pads its page 16 px above and below and 32 px
at the sides (`.app { padding: var(--size-4) var(--size-8) }` in its built
stylesheet), which the theme had left alone; and the bar's ground reached a
fixed 16 px into that padding, 16 px short of each edge on the real page. The
theme pads the page 8 px on every side now (`--mc-theme-page-pad`, the focus
root's inset, on `.gradio-container.app`) and the ground reaches by the same
variable. The layout test page had modelled Gradio's padding as 16 px all
round, which is why the short ground passed; it carries Gradio's own numbers
now, and `test_the_page_meets_the_window_as_focus_mode_does` measures the
content's edges against the window's in both modes, the ground's reach, and
Gradio's padding back when the switch is off.

**The sort buttons, and newest created first (2026-10-15).** "In the extra
networks, the sort buttons don't work! ... Also, default sort to newest
created to oldest." Forge renders each page twice -- empty when the UI is
built (`create_html(empty=True)`, which already carries the page's controls),
whole when the page loads (`interface.load`) -- and its script wires the
first render's controls into its strip, so every page keeps a second, hidden
copy of its controls under the same ids. Its sort reads the pressed sort
button with `gradioApp().querySelector('#<page>_controls
.extra-network-control--sort.extra-network-control--enabled')`, the first in
the document; Forge's own strip sits in the tab bar, before the pages, and
the drawer's head, holding that strip, had been appended after them, so the
sort read the hidden copy's button -- the default, never pressed -- and only
the direction, which Forge holds from when it wired the page, still turned
the list round. Reproduced with Forge Neo's own `extraNetworks.js` on the
layout test page: sorts with the theme off, does nothing in either layout
once the page has been rendered twice. The head now goes right after Forge's
tab strip (`placeHead`, moved only when out of place), a level above the
shown page in the drawer layout so its edge still runs down the page's side.
The default is Forge's own two settings, set once to Date Created and
Descending while both are still at Forge's defaults
(`mc_extra_networks_sort.default_newest`, a hidden record), and Forge's sort
keys are corrected so that `date_created` is the file's creation time where
the platform keeps one (Forge files the modification time there, and on
Windows the creation time under `date_modified`). And since Forge sorts a page
only when its own tab is selected, and its load renders the cards in the
listing's order, the drawer asks Forge's `applyExtraNetworkFilter` for the
page it shows once for each rendering of the page's cards (`sortShown`).
`tests/test_theme_layout.py` carries Forge's sort re-expressed and the hidden
second render; `tests/test_extra_networks_sort.py` the keys and the default.

**One scroller for the page (2026-10-17).** "When I scroll on text to image
or image to image view, the entire page scrolls, and in a weird way. The
generation button seems to scroll with the left side column. The two columns
scrolling independently seems to impact each other's positioning." Then the
rule asked for: the page takes one scroll action; the left column moves first,
down to its bottom, then the right column; the two never move at once;
Generate stays locked while the left column scrolls; and "scrolling needs to
now become a single master control that controls the two columns from one
master scroller" -- with the Forge Assistant's flyout still able to take the
left column, "so we will need handling to be completely independent for
gallery column and the configuration column." What the page had: the split
view's results column was `position: sticky` under the bar, and a sticky
element never leaves its containing block, the row, which was as tall as the
taller column. So the results held only while the settings were the taller;
both moved together over the last stretch, the infotext's height; with few
settings open or a long infotext the results never held and Generate
scrolled off with the page; and resting 88 px down with the hold at 80 the
column crept 8 px up to the bar's margin before holding (measured on the
layout test page, which has the user's shape). Built (`model_chain_theme.js`,
"one scroller for the page"; `model_chain_theme_scroll`, on): the master is
the page's own scroller -- the workspace's under focus mode -- and nothing
listens to it; the script measures three numbers and the stylesheet holds
the columns by them with sticky positioning, so the compositor moves them
(a script moving columns on scroll events would lag the finger on the user's
Gecko browser). The results column is held where it rests, never above its
place in the row (`--mc-theme-results-hold`, the greater of the bar's margin
and its place). The settings column is held once its bottom edge meets the
view's, by a sticky offset of the view's height less its own -- negative for
a column taller than the view; a shorter one is held where the results are
(`--mc-theme-scroll-hold`, `.mc-theme-scroll-settings`, a stacking context a
step above the results' so what the settings open over them is drawn over
them). And the row is given the least height the two phases need: the
settings column's plus the results' part under the view
(`--mc-theme-scroll-row`), so the results column is released exactly when
the settings end (`scrollPlan`, pure). Where the results are held is read off
the layout while they are held, because Chromium measures a sticky offset
from inside a scroller's padding and Firefox from its edge, and the settings
column's offset goes in the same convention. Measured again on every size
change of the row and the columns, every change of the body's classes (focus
coming and going changes the scroller without resizing anything), every
window resize and every apply, written only when changed. The scenarios: the
settings column docked in the assistant's panel is `position: fixed` by the
dock's rule, out of the row, scrolling inside the panel with its own
`overscroll-behavior: contain`; the script writes nothing for it, the row is
the results' height, nothing holds, and the page scrolls the results from
the first pixel; released, the two phases are back. Under focus mode the
workspace is the scroller and the phases are the same. A settings column
shorter than the view is held where it rests while the results scroll.
Stacked under the results on a phone, nothing is written. A **Details**
button at the end of the gallery's button row (`mc-theme-jump`, appended
once, its glyph rewritten only when the page's end comes or goes) takes the
scroller to the results' end and back to the top from there: the one price
of the order, one press instead of the flicks. `tests/test_theme_layout.py`
holds all of it in Chromium with the assistant's scripts on the page, on the
flex row and on Forge's resize-handle grid row, sampled every 97 px so that
each step's movement is one column's and adds up to the step; sixteen
mutations each fail a test. Not run on the user's machine or in Firefox.

**The user's first run (2026-10-17, the same day).** "This didnt work ... im
able to partially scroll the page, and instead of the left column moving
only, i see the right column move down and the generation button gets cut
off a bit." The screenshot showed the results column moving with the page
over its first fifty pixels, Generate going under the header. Not Gecko's
doing: a Gecko build (Zen, Firefox 158, driven through geckodriver; Mozilla's
own hosts are refused from this container) held the layout test page exactly
as Chromium does, with and without LibreWolf's fingerprinting defence and
letterboxing. The difference was the page, found by building Txt2Img in
Forge's shape on stock Gradio 4.40.0 with **Forge Neo's own `style.css`**,
read from its repository and served as Forge serves it, as `<link>` tags
(Gradio's `css=` prefixes every selector with its container class and
breaks a `:root` selector on a second line of a list): Forge makes
`#img2img_results, #txt2img_results, #extras_results { position: sticky;
top: 0.5em }` itself, and an id outranks the theme's
`:root[data-mc-theme="on"] [id$="_results"]`, so on the user's page the
column had been held half an em from the window's top, under the 64 px
header, since the theme was built -- "the generation button scroll off
page" in their first report was this rule, not the row's height -- while
the layout test page, which carried no such rule, held it under the bar.
The hold is written on the same two ids under the theme's attribute now
(1,2,0 against 1,0,0), and the test page carries Forge's rule, under which
the first build fails. Found beside it: Forge's `.unequal-height {
align-content: flex-start }` keeps a grid's one row track at its content
height, so the room `min-height` adds lies under the track; both engines
still hold a sticky grid item inside the grid container's content box
(measured with the rule reverted), but Forge's divider, `.resize-handle {
height: 100% }`, is as tall as the track and stopped at the settings
column's end through the second phase, so the least height goes on the
track as well (`grid-auto-rows: minmax(..., auto)`; the test page carries
Forge's divider rule and holds the divider to the row's height). And a
flaw of the first build that Gecko's run showed before the page did: a
measure taken in the second phase read the column's pushed position as its
hold and shrank the row -- the hold is the offset from the scroller's edge
now, with the browser's own addition to it read only while the column is
drawn held and kept per scroller (`entry.shift`), and a test measures in
the second phase and expects the same numbers. Checked on the Forge-shaped
Gradio page in Chromium and in the Gecko build: the column locked at its
resting place through the first phase, the two phases exact. Gradio's
container says `overflow: hidden`, under which nothing sticky in it could
hold against the page; Forge's stylesheet puts `overflow: visible
!important` back on it, which is why the theme's header sticks at all.

The pieces, and where they are:

- `mc_theme.py` — the **palette as a Gradio theme object** (the host's own
  mechanism: `gr.Blocks(theme=shared.gradio_theme)`; every attribute becomes a
  CSS variable, light and dark), installed on `on_before_ui` while the master
  switch is on and the host's own theme object put back when it is off; the
  **settings section** *Model Chain: Theme*; the **one-time import** of the
  user's file (newest `extensions/*/lobe_theme_config.json`; only a switch
  still at its default is written; the record in a hidden option, with the
  raw file, which a later build reads again for what the earlier one could
  not use — `_import_again`, the record's `version`).
- `javascript/model_chain_theme.js` — the page work, every piece a switch
  read live and undone when it goes off: Forge's own `#tabs > .tab-nav` as a
  64 px sticky **header** (never moved or wrapped: Forge reads the current tab
  from `#tabs > .tab-nav > button.selected`), with a title first, and last a
  quick-settings button, a **⋯ menu of the tabs that do not fit** (the
  selected one is always shown), a theme toggle and a settings button; the
  **split view** (the classic top row to the head of the settings column, the
  Generate box to the head of the gallery's container, in both tabs, with the
  compact layout's box handled too); **a generation without the Interrupt
  and Skip row** (`pressBar`: a press on the running bar or in what it paints
  above itself, never on a control, marks the bar to bring the row back,
  and the mark goes with the bar); the **quick-settings drawer** (fixed
  under the header, opened from the header or by the setting, closed by
  Escape, remembered per browser); the **extra-network drawer** on the right
  (Forge's `.extra-page` tabs and the controls strip Forge's script gathers
  moved into an aside with a strip of its own, one page shown at a time,
  Generation staying the selected Gradio tab; a page is taken only once that
  strip exists, because Forge wires the pages after their HTML arrives and
  looks for them as its own tabs' children; Forge's tab strip is hidden
  under a class the script puts on it, never by a selector naming tab bars);
  the **generation-info panel** (Forge's `html_info_*` paragraph read through
  its `<br>`s and parsed as `infotext_utils` parses it — a last line of three
  pairs or more is the parameters, a quoted value keeps its commas — drawn as
  blocks and chips beside the paragraph, which the stylesheet clips out of
  view; redrawn when the markup changes); **resizable prompts**; **icons**
  from the ISC-licensed Lucide set (the notice is in the file); optional
  **tab names with spaces** ("Txt 2 Img", but "LoRA" and "WanGP" stay);
  **ask before leaving**; the **typeface** link. It **stands down** while a
  Lobe root is on the page (`data-mc-theme="lobe"`), and marks the root from
  the last configuration at the first paint so a themed page is themed before
  the host has said it is loaded.
- `style.css`, *Model Chain -- the theme* — keyed on `data-mc-theme="on"`:
  the header, the menu, the two drawers (the extra-network one with its
  strip, its page at the card size set, its pane as tall as its cards), the
  info panel's blocks, chips and copy buttons, the split view's rows, the
  sticky results column, a running progress bar in the results panel's flow
  at 14 px with the Generate box not drawn beside it (`:has()`, keyed on a
  bar Forge has not hidden and the press's mark), the boxes down the settings column, the
  token-counter pill, the 36 px tool buttons, the gallery's checkerboard
  ground, the phone sizes. Colours only from the host's variables;
  `!important` only where Forge, Gradio's inline style on a tab panel or a
  card, or a theme's rule about every button says it first.
- Tests: `tests/test_theme.py` (settings, palette, theme object, install,
  import, the record read again), `tests/test_theme_js.py` (the pure parts in
  node, the infotext parser, the stylesheet as text),
  `tests/test_theme_layout.py` (Chromium: the bar, the overflow, the split,
  both drawers — the extra-network one against Forge's own markup, with the
  pages wired late as Forge wires them — the info panel, the prompts, the
  icons, standing down, off, the first paint, a phone width).

How the user moves over: Settings → Model Chain: Theme → *Use the Model Chain
theme* on, Apply, Reload UI (the imported switches are already there); then
Extensions → disable `sd-webui-lobe-theme`, Apply and restart; the page is
the same page, drawn by this extension. Lobe's folder is never written to.

The analysis below is kept as the record of what was found in the Lobe
repository and how Forge loads a theme, which is what the rebuild was designed
from; its plan is not.

---

## 1. What the Lobe extension is made of

| | |
|---|---|
| `javascript/main.js` | 2.2 MB. A Vite build of `src/` (React 18, antd 5.17.0, `@lobehub/ui`, antd-style/emotion, i18next, zustand). A **classic script** — no top-level `import` or `export`, which is why Forge's plain `<script src>` can load it — with one dynamic chunk, `javascript/giscus-BNK3dBIH.js`, pulled in by `import("./giscus-BNK3dBIH.js")` relative to the bundle's own URL when the feedback dialog opens. |
| `scripts/settings.py`, `scripts/lib/*.py` | On `on_app_started`, five routes: `GET /lobe/package` (its `package.json`, for the version shown), `GET /lobe/prompt` (`data/prompt.json`, the prompt formatter's library), `GET /lobe/locales/{lng}` (`locales/*.json`, i18n), and `GET`/`POST`/`DELETE /lobe/config` — **the user's settings**, as `lobe_theme_config.json` in the extension's own folder (git-ignored there). |
| `preload.py` | One launch flag, `--lobe-debug`, which `scripts/lib/lobe_log.py` reads at import as `shared.cmd_opts.lobe_debug`. |
| `package.json`, `data/`, `locales/` | Read by the routes above. |
| `src/` and the build files | The TypeScript source, `vite.config.ts`, `tsconfig.json`, lint and release configuration. The release workflow does **not** build: the committed `javascript/main.js` is built by hand. |
| `LICENSE` | **AGPL-3.0** (the file is AUTOMATIC1111's licence text reused; `package.json` says `AGPL3`). |

### What the bundle does to the page

Read from `src/`, and already modelled by this repository's tests
(`tests/test_assistant_js.py`'s `lobify`, `tests/test_assistant_focus_js.py`,
`tests/test_assistant_layout.py`'s host CSS, `tests/test_llm_studio_js.py`'s
`LOBE_SWAPS`):

1. On `DOMContentLoaded` it appends `<div id="root">` to `gradio-app` and
   renders a React tree into it: a 64 px sticky header with its own tab
   strip, two sidebars, a content column and a footer, with a loading screen
   until `onUiLoaded` has fired and its store has loaded.
2. Then it **moves Forge's own elements** into that tree (`useInject`):
   `.app` into the content column, `#footer` into its footer, `#quicksettings`
   into the left sidebar, the extra-networks panels into the right. With
   *split previewer* on it also moves `#txt2img_toprow` into
   `#txt2img_settings` and `#txt2img_generate_box` into the gallery's
   container (which is why Generate sits above the gallery in the user's
   screenshot), and the same for img2img and Extras.
3. It injects about fifteen stylesheets through antd-style into `<head>`:
   Gradio's own variables redefined on `:root, .dark` from antd tokens
   (`src/styles/tokens.ts`), and rules for buttons, inputs, galleries, tabs,
   dropdowns, the lightbox, other extensions' panels, and a mobile layout.
   The accent is `token.colorPrimary`, **white in night mode and black by
   day** unless a primary colour is chosen in its settings; that white is the
   Generate button the user asked about.
4. It puts `dark` or `light` on `<body>` (from `?__theme=` or
   `prefers-color-scheme`), runs `replaceIcon` once at mount (the SVG icon
   swap this repository's CLAUDE.md warns about), and installs the prompt
   editor, the prompt formatter, the share dialog and the image-info overlay.
5. Off the page it talks to: the five `/lobe/*` routes (local); GitHub's API
   for the latest release (the version tag in the feedback dialog); a CDN for
   its favicons and, with *web fonts* on, its typefaces; Giscus when the
   feedback dialog opens.

### The settings

`src/store/initialState.ts` declares the whole set, 27 keys:
`confirmPageUnload`, `enableExtraNetworkSidebar`, `enableHighlight`,
`enableImageInfo`, `enableSidebar`, `enableWebFont`, `extraNetworkCardSize`,
`extraNetworkFixedMode`, `extraNetworkSidebarExpand`,
`extraNetworkSidebarWidth`, `i18n`, `layoutHideFooter`, `layoutSplitPreview`,
`liteAnimation`, `logoCustomTitle`, `logoCustomUrl`, `logoType`,
`neutralColor`, `primaryColor`, `promptEditor`, `promptTextareaType`,
`sidebarExpand`, `sidebarFixedMode`, `sidebarWidth`, `svgIcon`.

Where they live, in the order the bundle reads them (`src/store/action.ts`):

1. the server, `GET /lobe/config` — the file `lobe_theme_config.json`;
2. the browser, `localStorage["SD-LOBE-SETTING"]`;
3. the browser again, the older `localStorage["SD-KITCHEN-SETTING"]`;
4. the defaults.

Whatever it finds is merged over the defaults and **posted straight back to
the server**, and every change is written to both the server and
`localStorage`. So the one file is the whole of "my settings", and every
browser that has used the theme carries a copy of its own. There is no other
server-side state: the prompt library and the locales are read-only data that
ship with the extension.

### Quirks worth knowing before vendoring

- **No lockfile.** `.npmrc` says `lockfile=false`, `.gitignore` ignores
  `bun.lockb` and `package-lock.json`, and `resolution-mode=highest`. A rebuild
  from source today resolves today's `@lobehub/ui` 1.x and friends, not the
  versions the committed bundle was built from in May 2024, and would not
  reproduce the bundle byte for byte. The committed `main.js` is the artefact
  the user actually runs.
- **`process.env` is inlined whole.** `vite.config.ts` has
  `define: { 'process.env': process.env }`, so the build machine's environment
  is in the bundle; in this copy only `NODE_ENV: "production"` is visible, but
  it is a reason to prefer the committed build over one made on a machine
  with secrets in its environment.
- **A URL containing `dev` switches the theme off.** `src/main.tsx` skips
  loading when `window.location.href.includes('dev')` in a production build,
  and the built bundle carries it: `href.includes("dev")&&!0||document.addEventListener(...)`.
  A Forge reached as `https://dev.something/` or with `dev` anywhere in its
  query shows no theme and says nothing.
- **Gradio 3 hashes.** `src/styles/components/label.ts` and others target
  `.svelte-1ojmf70`, `.svelte-1p4r00v`, `.svelte-awbtu4`, `.svelte-j1gjts`:
  Gradio 3-era classes that match nothing on 4.40. Dead weight, harmless.
- **Upstream is quiet.** 3.5.4 is from 2024-05-24; the version check against
  GitHub reports "latest" as long as that holds, and would nag the moment
  upstream tags a release the vendored copy does not have.

---

## 2. How Forge loads it — and why the host can be anyone

From Forge Neo's own code:

- `modules/extensions.py`, `list_extensions()`: the extension folders in
  `sorted(os.listdir(...))` order, which is case-sensitive, so on this
  machine the order is `SD-Neo-ModelSwitchRefiner`, `a1111-mini-paint-NEO`,
  `sd-webui-lobe-theme`. An extension is enabled unless its folder name is in
  `shared.opts.disabled_extensions` (or Forge's own always-disabled list).
- `modules/scripts.py`, `list_scripts("javascript", ".js")`: the files **at
  the top** of each enabled extension's `javascript/` folder (no recursion),
  sorted by filename within the extension, extensions in the order above, then
  a topological sort on any `metadata.ini` `Before`/`After`/`Requires` lines.
- `modules/ui_gradio_extensions.py`: one `<script type="text/javascript"
  src="file=<path>?<mtime>">` per file, written into `<head>` after Forge's
  own `script.js`; one `<link rel="stylesheet">` per enabled extension's
  top-level `style.css`, then `user.css`, written before `</body>`. Both go
  through the `file=` route, which serves anything under the extension's
  folder — the giscus chunk's relative import works for the same reason.

So the Lobe extension is, to Forge, a folder with one script and five routes.
A bundle with the same bytes in any enabled extension's `javascript/` folder
is the same theme, and the only things that tie the bundle to its folder are:

1. the five `/lobe/*` routes, served by whoever registers them — FastAPI keeps
   the first registration of a path, and this extension's `scripts/` load
   first by the order above;
2. `giscus-BNK3dBIH.js`, which must sit beside the bundle;
3. `package.json`, `data/prompt.json` and `locales/*.json`, read by the routes.

---

## 3. Why Mini Paint NEO needs no change

Checked by reading both repositories for anything that names the Lobe
extension, its folder, or its routes:

- Mini Paint NEO names neither `sd-webui-lobe-theme`, nor `/lobe/`, nor
  `lobe_theme_config` anywhere outside test stand-ins
  (`tests/test_wangp_protocol.py` serves a stylesheet from a made-up
  `extensions/lobe/` path to test cross-origin font handling). This
  repository names the folder only in test docstrings and CLAUDE.md.
- What Mini Paint depends on is **the bundle's behaviour**, read at runtime:
  the page's CSS variables (`--font`, `--font-mono`, the text, page and
  accent colours its WanGP theme hijack samples), the `.dark` class on the
  body, Lobe's `button { min-width: fit-content !important }` (which its
  Clipboard tiles pin against), the sticky results column and 64 px header
  its layout tests carry, the `visibility: hidden` reading of a parked panel,
  and this extension's `forge-assistant-focus-root`. All of it is the same
  bundle doing the same things from a different folder.
- **Load order.** Hosting moves the bundle from the last extension's slot to
  the first's. The bundle does nothing at evaluation but register a
  `DOMContentLoaded` listener (and spill its top-level `var`s onto `window`,
  which it already does; Mini Paint's bundles are closures and read none of
  them). Its React mount still happens at `DOMContentLoaded`, its moving of
  Forge's elements still happens in a microtask after the `onUiLoaded`
  callbacks have all run (React 18 batches a state change made outside a
  React event), so nothing Mini Paint registers with `onUiLoaded` sees a
  different page than it does today. Mini Paint's bundles register no
  `DOMContentLoaded` listener of their own except one in
  `browser/minipaint_wangp.js`, which does not look for Lobe's tree.

What *would* break Mini Paint is a different bundle: one rebuilt from source
with today's dependencies could draw a different `button` rule, a different
header height, different variables. That is the argument for vendoring the
committed build as it is (section 4) and making every change of our own as a
layer on top (section 6), which is also what the night mode already is.

---

## 4. The design

### 4.1 Vendor the build, keep the source beside it

- `theme/lobe/` — upstream's tree as reference: `src/`, `locales/`, `data/`,
  `package.json`, `vite.config.ts`, `tsconfig.json`, `LICENSE`, `README.md`,
  `CHANGELOG.md`; no `node_modules`, no build here.
- `javascript/lobe_theme.js` and `javascript/giscus-BNK3dBIH.js` — the built
  bundle and its chunk, **byte for byte the user's 3.5.4** apart from the one
  anchored patch in 4.4, so that what the user runs after the move is what
  they run now. (`lobe_theme.js` sorts among this extension's own scripts,
  between `llm_studio.js` and `mc_message_editor.js`; nothing here depends on
  evaluation order, every script waits for the page.)
- `tools/vendor_lobe_theme.py <zip or folder>` — copies the above into place,
  applies the patch, records upstream's SHA-256 and version in
  `theme/lobe/VENDORED.json`, and refuses a bundle in which the patch's
  anchor occurs other than exactly once. A test holds that the vendored
  bundle carries the patch and the recorded hash matches the file.

### 4.2 The routes

`mc_theme_api.py`, installed on `on_app_started` like `mc_voice_box_api`:

| Route | Answer |
|---|---|
| `GET /lobe/package` | the vendored `package.json` |
| `GET /lobe/prompt` | the vendored `data/prompt.json` |
| `GET /lobe/locales/{lng}` | the vendored `locales/<lng>.json`, `{"error": …}` when missing, as upstream |
| `GET /lobe/config` | the settings file, or `{"empty": true}` with 404 when there is none (the bundle then falls back to the browser's copy) |
| `POST /lobe/config` | writes the settings file |
| `DELETE /lobe/config` | removes it |

The settings file lives at `<WebUI data path>/model_chain_theme/lobe_theme_config.json`
— under the data directory like every other thing this extension keeps, never
inside the checkout (the test guard forbids writes there, and an update of the
extension must not lose the theme's settings the way updating Lobe's own
folder could). `--lobe-debug` is registered by a `preload.py` here, or read
with a `getattr` default, so a launch script that still passes it keeps
working.

The route names stay `/lobe/*` on purpose: the bundle asks for them by those
names, nothing else in the WebUI or in Mini Paint does, and keeping them is
what lets the bundle stay unpatched there.

### 4.3 Importing the user's settings

On the first start with no settings file of ours, `mc_theme_api` looks for
`lobe_theme_config.json` in every folder under `extensions/` that looks like
the Lobe extension — identified by content, not by name: a `scripts/lib/config.py`
that names `lobe_theme_config.json`, or a `javascript/main.js` that asks for
`/lobe/config` — enabled or disabled, and copies the newest. One console line
says what was imported from where. Nothing is deleted from the original.

That covers the user's machine. The bundle's own fallback covers everything
else: with no server file it loads `localStorage["SD-LOBE-SETTING"]` and posts
it straight back to `/lobe/config`, so a browser that has used the theme
restores the settings by itself on its first load after the move, even if the
original folder was removed before this extension ever ran.

### 4.4 One load where both could mount

While the original extension is still enabled, two copies of the bundle are on
the page and each would append a `#root` and render — two headers, Forge's
elements moved into whichever rendered first, and no clean way back. The
hosted copy has to stand down while the original is present, and the decision
has to be made **before `DOMContentLoaded`**, because that is when the bundle
mounts; nothing Gradio renders exists yet, and `opts` is not published until
later.

Two ways, both workable:

- **A loader with a cached decision (recommended).** The bundle is not in
  `javascript/` at all but at `theme/lobe/javascript/main.js`, and a small
  `javascript/lobe_theme_loader.js` decides synchronously from
  `localStorage["mc-theme-host"]` whether to `document.write` the bundle's
  `<script>` tag (same-origin, so no browser intervention applies), then
  fetches `/model-chain/theme/state` and stores the server's current answer
  — host or not, and whether an original Lobe extension is enabled — for the
  next load. No cache means *do not host*: on the very first load after
  installing, the original (still enabled) draws the page exactly as before
  and nothing changes; once the user disables it, the next load's cache says
  host and the hosted copy takes over. A decision that changes takes effect
  on the next load, which is how Lobe's own settings dialog already works
  (it reloads the page on save). The one window is a load straight after
  re-enabling the original, when a stale cache lets both mount once.
- **A patched mount.** Keep the bundle in `javascript/` and replace, at
  vendoring time, `document.addEventListener("DOMContentLoaded",` with
  `(window.mcThemeHost||document).addEventListener("DOMContentLoaded",`;
  a guard script that sorts before it provides `mcThemeHost`, whose listener
  checks `document.scripts` at `DOMContentLoaded` for another
  `/javascript/main.js` from a folder other than ours and stands down if
  there is one. Deterministic on every load, at the price of a (one-line,
  anchored, tested) patch to the bundle.

The loader also answers the question of **other users of this extension**: a
theme they did not ask for must not appear because they installed an LLM and
voice extension. `model_chain_theme` (none / Lobe) defaults to *none*; the
user flips it once, and the loader's cache does the rest. Settings → Model
Chain shows one read-only line: which theme is hosted, where its settings file
is, what was imported, and — while the original is still enabled — that the
hosted copy is standing down and why.

### 4.5 What the user does

1. Update this extension; start Forge; nothing changes (the original still
   draws the page; the settings file is imported in the background).
2. Settings → Model Chain → *Theme*: Lobe. Disable `sd-webui-lobe-theme` in
   the Extensions tab (or delete its folder). Reload.
3. The same page, drawn by the hosted copy; every setting as it was; Lobe's
   own settings dialog still works and now writes to our file.

---

## 5. Risks and limits

- **Updates stop flowing by themselves.** Upstream has been quiet since
  2024-05-24, and re-importing a new release is one run of the vendoring
  tool; but a Gradio or Forge change that breaks the theme is now this
  repository's to handle. That is the price of being the host.
- **The `dev` URL quirk** and the GitHub version nag stay as upstream built
  them unless the bundle is patched or rebuilt (section 6).
- **Two bundles for one load** in the window described in 4.4.
- **The licence** (section 8).
- **Nothing has run on the user's machine**, as with everything in this
  repository until it has.

---

## 6. Changing the theme itself

Three layers, from cheapest to most involved:

1. **This extension's stylesheet and scripts over the bundle** — what the
   night mode is. The bundle stays the user's bytes; the change is a Forge
   setting with a test that measures it in a real browser. Right for colour,
   spacing and small behaviour.
2. **Anchored patches at vendoring time** — one-line, counted, tested text
   replacements in the built bundle, for things a stylesheet cannot reach: the
   mount gate, the `dev` quirk, the version check. Right while they stay few.
3. **A reproducible build from source** — a lockfile made with the
   dependency versions of May 2024 (`npm install --before=2024-05-25` resolves
   to what was published before that date) and a check that the result
   behaves as the committed bundle does, after which `src/` is ours to edit
   and `tools/build_theme.py` makes the bundle. Right once there is a change
   worth making in React — a settings page of our own, the icon swap taught
   not to eat a span's whole content, the CDN calls removed.

The night mode would move from layer 1 to layer 3 only if the user wants it
*in* the theme rather than over it; nothing requires that.

---

## 7. Phases

| Phase | What | Where it stands |
|---|---|---|
| 0 | A quieter night mode: Generate and every primary button as a dim, accent-tinted surface in the dark theme; a setting, on by default; measured in Chromium | **built**, this session |
| 1 | Vendor 3.5.4 (bundle, chunk, source, data, licence), the five routes, the settings file under the data path, the import from the original folder, the loader with its cached decision, the Settings line, the vendoring tool and its tests | next session |
| 2 | The user migrates (4.5); then the original folder goes; `tests/` stand-ins keep modelling the bundle as they do now | with the user |
| 3 | Optional: the reproducible build, and whatever theme changes the user wants inside the theme | when asked |

---

## 8. The licence — the one thing to decide first

`sd-webui-lobe-theme` is **AGPL-3.0**. Copying its source and build into this
repository is allowed by that licence on its terms: the copy keeps its
copyright and licence notices, and the combined work, when distributed, is
itself under the AGPL-3.0 with its source available. This repository carries
**no `LICENSE` file today**, which in law is *all rights reserved*, and
redistributing an AGPL work inside it under that would not be permitted.
Mini Paint NEO is MIT and is a separate work; nothing there changes.

So vendoring means one of:

- adding an `AGPL-3.0` licence to this repository (the simplest honest
  position: it is a Forge extension, Forge itself is AGPL-3.0, and so is the
  theme);
- or keeping the Lobe extension as a **dependency** rather than a part — a
  `git submodule` or a pinned download into the data path at install time —
  which keeps the two works separate at the cost of the "one folder, one
  extension" simplicity the user asked for.

The first is recommended; it is the user's call, and nothing in phase 1 is
started until it is made.

---

## 9. What was read, and what was not

Read in full: `preload.py`, `scripts/settings.py`, `scripts/lib/*.py`,
`src/main.tsx`, `src/app/*`, `src/layouts/index.tsx`, `src/store/*`,
`src/styles/*` and `src/styles/components/*`, `src/features/Header/*`,
`src/features/Content/*`, `src/features/Footer/index.tsx`,
`src/features/Setting/{index,data}.tsx` and `Form/Appearance.tsx`,
`src/hooks/useInject.ts`, `src/scripts/replaceIcon.ts`, `src/const/url.ts`,
`vite.config.ts`, `package.json`, `.npmrc`, `.gitignore`, `.github/workflows/*`.
Searched, not read: the 2.2 MB bundle, for its routes, storage keys, dynamic
imports, its mount, the inlined environment and the `dev` quirk.

Not read: the sidebars', share, prompt-formatter and image-info features in
detail (they move and decorate Forge's elements and need nothing from a host),
the locales, the prompt library.
