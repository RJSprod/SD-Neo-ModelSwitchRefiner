# Item 6 — Optional progress-bar styling

Project: SD-Neo-ModelSwitchRefiner ("Model Chain")
Host: Stable Diffusion WebUI Forge Neo (`neo`)

**Revision 2.** Rewritten after reading the host's `javascript/progressbar.js`
and `style.css`, and Lobe Theme's `src/styles/components/progress.ts`. Revision 1
guessed at the constraints; this one names them.

Implemented in `style.css` and `javascript/model_chain_progress.js`.

---

## Background

This item is deliberately separate from Item 4. Styling changes how the host's
bar looks and never what it reports; the calculation changes what it reports and
never how it looks. Turning either off leaves the other working.

The separation is literal in the implementation: the styling layer is CSS and
one JS file with no Python of its own, and the calculation is Python with no DOM
of its own. Nothing crosses.

---

## How the settings reach the browser

Every option registered through `shared.options_templates` arrives in the
browser as a field of the global `opts` object — the same mechanism
`progressbar.js` uses for `opts.show_progressbar`. So the styling layer needs no
endpoint, no Gradio component and no Python. It reads `opts` on
`onOptionsAvailable` / `onOptionsChanged` / `onUiLoaded` and writes classes and
one custom property onto `document.documentElement`.

| Setting | Effect |
| --- | --- |
| `model_chain_style_enable` | master toggle; adds `.mc-progress-styled` |
| `model_chain_style_theme` | named look; see below |
| `model_chain_style_color` | any CSS colour, including `rgba()`; empty follows the theme |
| `model_chain_style_gradient` | Custom theme only: fade towards the leading edge |
| `model_chain_style_sheen` | Custom theme only: travelling highlight |
| `model_chain_style_glow` | Custom theme only: outer glow and pulse |
| `model_chain_style_complete` | one flash when a job truly finishes |

A colour the browser does not understand is rejected by `CSS.supports('color', …)`
and logged once, rather than written to the variable where it would blank the
fill.

---

## Themes

The stylesheet implements four independent effects — `gradient`, `sheen`,
`glow`, `pulse` — plus an `intense` modifier, each exactly once, keyed on a
`.mc-fx-*` class. **A theme is a choice of those effects, made in the JS.**

| Theme | gradient | sheen | glow | pulse | intense |
| --- | :-: | :-: | :-: | :-: | :-: |
| Flat (default) | | | | | |
| Gradient | ● | | | | |
| Sheen | ● | ● | | | |
| Pulse | | | ● | ● | |
| Neon | ● | ● | ● | | ● |
| Custom | *toggle* | *toggle* | *toggle* | *with glow* | |

Two consequences of putting the themes in the JS rather than giving each one its
own CSS:

- There is one implementation of each effect to get right, not one per theme. A
  rendering problem is fixed in one place.
- `Custom` is not a special case. It resolves to the same capability classes by
  a different route.

Python contributes only the dropdown's choice list. A test parses the JS and
asserts the two agree, so a theme cannot be offered that nothing renders.

**Every theme derives its colours from one custom property**, `--mc-progress-fill`,
through `color-mix`. That is what makes the colour setting orthogonal to the
theme choice — one colour recolours whichever look is selected, rather than only
the plain one. A solid `background-color` is declared before any gradient so a
browser without `color-mix` still gets a correctly coloured bar.

`prefers-reduced-motion: reduce` disables the sheen and the pulse and suppresses
the completion flash entirely.

---

## Constraints found in the host and in Lobe

These are the reason the design looks the way it does. All three were verified
against source, not assumed.

### 1. The bar's text is rewritten twice a second

`progressbar.js` sets `divInner.textContent = progressText` on every poll. Any
child element injected into `.progress` is wiped within `live_preview_refresh_period`
(500ms by default). **Nothing in the styling layer relies on child nodes.**

### 2. `::before` and `::after` on `.progress` are already taken

Lobe Theme puts an animated diagonal stripe on `::before` and a top highlight
gradient on `::after`. Claiming either would *replace* the theme's effect rather
than layer with it.

**Resolution: the styling layer uses no pseudo-elements at all.** The gradient
and the travelling sheen are stacked `background-image` layers on the element,
animated through `background-position`; the glow is `box-shadow` on the element.
This removes the whole collision class rather than detecting and working around
it — and `box-shadow` has the separate virtue of spreading *outside* the bar, so
it cannot reduce the contrast of the percentage and ETA text sitting on the fill.
That answers revision 1's watchpoint about animation making the text harder to
read.

There is one remaining interaction, and it is visual rather than structural: a
theme animating its own overlay plus our sheen puts two moving patterns on one
small element, which reads as a fault. The JS measures the actual conflict —
`getComputedStyle(bar, '::before').content` on the live bar — and drops the
sheen when it finds one, leaving colour and glow to carry the look. Measuring
the conflict rather than recognising the theme means it holds for any theme that
decorates the bar the same way, including ones that do not exist yet.

`animation` is a single shorthand, so sheen and pulse together would replace
rather than compose. They drive different properties and are emitted as one
two-name declaration by a dedicated rule.

### 3. Themes move the bar

Forge's own `style.css` puts `.progressDiv` at `position: absolute; top: -14px`.
Lobe overrides it to `position: relative !important; top: 0 !important`, taking
it out of the overlay and into normal flow.

**Nothing in the styling layer sets geometry** — no height, no offset, no radius.
The bar stays wherever its theme puts it and only its colour and animation
change.

The completion flash carries the host's own `progressDiv` / `progress` class
names alongside its own, so it inherits whatever geometry the active theme gives
the real bar. That is what stops it shifting the layout under a theme that has
moved the bar.

---

## Lobe Theme specifically

Lobe does **not** replace the progress bar. It is React, but `progress.ts`
styles the native `.progressDiv > .progress` DOM. There is no component to
intercept.

**Cascade.** Lobe injects through antd-style's `createGlobalStyle`, which emits
unprefixed global selectors into `<head>` at mount. Forge injects extension
`style.css` as a `<link>` before `</body>` (`modules/ui_gradio_extensions.py`),
which is later in document order, so equal-specificity ties go to the extension.
The rules here are `.mc-progress-styled .progressDiv .progress` (0,3,0), one
level above both Forge's and Lobe's, so no `!important` is needed anywhere.

**Fill colour is uncontested.** Lobe styles the `.progressDiv` *track* with
`!important` but never sets a background on `.progress` — the fill still comes
from Forge's `.progressDiv .progress { background: #0060df }`. Under Lobe today
the bar is therefore Forge blue with Lobe's stripes over it, ignoring Lobe's own
accent.

**Safe default: `var(--primary-500)`.** Gradio themes define `--primary-50`
through `--primary-950`; Lobe redefines exactly those against its geekblue ramp
(`src/styles/tokens.ts`). So an untouched install follows the active theme's
accent in light and dark and under either UI — and under Lobe it is an
improvement on the current appearance rather than a change to it. This is also
how revision 1's "respect light/dark themes and provide safe defaults" is met:
by inheriting the theme's own tokens rather than by branching on a theme name.

**Unverified.** Lobe's README claims SD WebUI ≥ 1.6 and does not mention Forge;
Forge Neo's extension-compatibility discussion does not list it; some of Lobe's
selectors carry Gradio-version-specific hashes (`.wrap.svelte-j1gjts`). Nothing
here depends on Lobe being present or absent — the accommodations are structural
(no pseudo-elements, no geometry, theme tokens for defaults) rather than
conditional on detecting it.

---

## The completion effect

`removeProgressBar()` deletes `.progressDiv` the instant `res.completed` arrives,
so there is nothing left to flash. `requestProgress` is a plain global function
declaration and its callers look it up by name at click time, so the styling
layer replaces the global, wraps the `onProgress` and `atEnd` callbacks, and
inserts its own element when the bar goes away.

**"Must not fire at the end of Stage 1" is satisfied structurally.** Stage 2 runs
inside `postprocess` of the same task, so there is exactly one progress-bar
lifecycle per Generate click. The bar is never created or removed mid-chain,
which also settles "no duplicate progress bars".

**Interruption is a heuristic, and this is a known limit.** `/internal/progress`
reports an interrupted task as `completed`, exactly like a finished one — the
host does not distinguish them on this channel. The flash fires only if the
highest progress seen reached 95%, so an interrupt early or mid-job is silent
and one at 97% is not. The alternative would be a second endpoint polled purely
to classify the ending, which is not worth it for a decorative flash.

The flash element removes itself on `animationend`, with a timeout backstop for
the case where the animation never runs, so nothing persists after completion or
interruption.

---

## Accessibility

`prefers-reduced-motion: reduce` disables the pulse and suppresses the flash
entirely. The glow is drawn outside the fill so the percentage and ETA text keep
their contrast in every combination.

---

## Acceptance

- [x] Styling can be enabled and disabled without touching Model Chain, and
      vice versa.
- [x] A small set of ready-made themes is offered, and a Custom one that defers
      to the individual toggles.
- [x] The colour setting is orthogonal to the theme choice — it recolours every
      theme, not only the plain one.
- [x] Colour customisation applies to the ordinary Forge progress bar, with
      Model Chain inactive or uninstalled from the generation.
- [x] `rgba()` and every other CSS colour form is accepted; invalid input falls
      back to the theme accent rather than blanking the bar.
- [x] The completion effect fires once at true completion, and cannot fire at
      the end of Stage 1.
- [x] Ordinary Forge generations keep their normal calculation and can use the
      custom look.
- [x] Chained generations use Item 4's calculation and the same style.
- [x] No duplicate bars; no animation persists after completion or interruption.
- [x] Light and dark are respected by default, through the theme's own tokens.
- [x] Works with Lobe Theme's restyled bar without fighting it.

## Relevant code

- Model Chain: `style.css`, `javascript/model_chain_progress.js`,
  the appearance settings in `scripts/model_chain.py`
- Forge Neo: `javascript/progressbar.js`, `style.css`,
  `modules/ui_gradio_extensions.py`
- Lobe Theme: `src/styles/components/progress.ts`, `src/styles/tokens.ts`
