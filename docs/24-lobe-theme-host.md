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
