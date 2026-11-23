# The settings as accordions, and the Settings Manager (2026-11-23)

> "Please screenshot in this message (please disregard the tab bar row, thats a
> screen shot glitch). For one, i want everything that is not prompt (positive,
> negative, literal, and lora viewer), to be in an accordion. Hi res already
> has it, but generation config like batch and resolution should be collapsible
> as well. Honestly, i would like a last item added to this list that is a
> "Settings Manager", which would have options for me to rearrange or show/hide
> the other accordions in real time."

The screenshot was Txt2Img's settings column on a phone under the Model Chain
theme: the Literal boxes, the compact style's tool row and styles, Sampling,
Hires. fix, Width/Height/Batch, CFG, Seed, an empty box (Override Settings),
then the scripts' accordions.

`javascript/model_chain_settings_manager.js`, its section of `style.css`
(*Model Chain -- the settings as accordions, and the Settings Manager*), and one
setting, `model_chain_settings_manager` (Settings → Model Chain: Page, on).

## 1. What Forge Neo draws in the column

Read from Forge Neo's `modules/ui.py` (`create_ui`, the categories in
`ordered_ui_categories`), `ui_toprow.py`, `ui_components.py` and the processing
scripts it places by section (`sampler.py`, `seed.py`, `refiner.py`), and
measured on a real Gradio 4.40 page built from them, with Forge Neo's own
`style.css`, `script.js`, `inputAccordion.js` and `token-counters.js`:

| Category | What lands in `#<tab>_settings` |
|---|---|
| prompt (compact style) | `#<tab>_prompt_container` (the prompt rows) and a row `.toprow-compact-stylerow` (the tool row `#<tab>_tools`, the styles row) |
| image (Img2Img) | `#mode_img2img` (the image tabs) and a row holding `#resize_mode` |
| inpaint (Img2Img) | `#inpaint_controls`, a group Gradio hides off the Inpaint tab |
| sampler | the sampler script's group (`create_group`): `gr-group > .styler > #sampler_selection_<tab>` |
| accordions | `#<tab>_accordions`, a row: each InputAccordion's hidden checkbox in a hidden form, then its accordion (Hires. fix `#txt2img_hr`, Refiner) |
| dimensions | a row of `#<tab>_column_size`, the swap button's column and, together with it by default, `#<tab>_column_batch` |
| cfg | a row: the CFG sliders and the cfg-section scripts |
| denoising (Img2Img) | the bare slider `#img2img_denoising_strength`, in a form Gradio puts round it |
| seed | `#<tab>_seed_row` and the group `#<tab>_seed_extras` (hidden until Extra) |
| batch | `#<tab>_column_batch`, when not with the dimensions |
| override_settings | `#<tab>_override_settings_row`, whose only form is hidden until a paste fills the dropdown -- the empty box in the screenshot |
| scripts | `#<tab>_script_container` (`gr-group > .styler`): a group per always-on script, each holding its accordion, then the Script choice's form, then a hidden group per selectable script |

The classic prompt-box style builds `#<tab>_toprow` above the tab instead; the
Model Chain theme's split view moves it into the column, first.

## 2. Decisions

| Question | Decision |
|---|---|
| Where | Txt2Img and Img2Img, everything in the settings column after the prompt area (`#<tab>_toprow`, `#<tab>_prompt_container`, anything holding the Prompt): one list, `.mc-sm-list`, put after the prompt area's last element. The prompt boxes, the Lora Viewer and the Literal row are untouched |
| What is a section | Each child of the column, and each child of a container walked into: the accordions row, the script container's `.styler`, a form Gradio put round several components. A piece that is an accordion -- itself, or the one thing a script's group draws -- is a section as it is; a piece Forge names (§1) is a section under that name, consecutive pieces of one name one section (the seed's row and its extras, the image tabs and Resize Mode); anything else that draws is a section named by its first label; a piece that draws nothing and is not Forge's rides with the section after it (an InputAccordion's hidden checkbox, a script group with nothing visible). Everything after the Script choice in the script container is the Script section's: the chosen script's own controls |
| Names | Tools & Styles, Image, Inpaint, Sampling, Size (Size & Batch with the batch in it), Batch, CFG, Denoising, Seed, Override Settings, Script; an accordion's own label |
| Keys | The Forge names above; an accordion or block by its id unless Gradio made it up (`component-N`, `input-accordion-N`), else by its label. Per tab |
| The new accordions | Built of the classes of one of Gradio's accordions on the page, read at build -- its generated `svelte-…` classes included, because Accordion.svelte's rules are scoped to them -- with Block.svelte's inline border and overflow, so the page's theme draws them as Gradio's (on a real page: the same padding, border, radius and background as Hires. fix, measured). A page with no accordion to read gets `.mc-sm-own`, drawn from the theme's block variables. Open until closed; a press on the label opens or closes it |
| Existing accordions | Moved as they are, their open state never written: Forge's InputAccordion switches Hires. fix on when its accordion opens |
| The manager | The last item, an accordion of the same kind, closed until opened: a row per section with a grip (a pointer drag captured on the list, never on the moving row; the arrow keys on the grip), ↑, ↓ and an eye (`aria-pressed`). Every change applies to the column as pressed. Reset: Forge's order, every section shown and open |
| Kept | Per browser and tab, `localStorage["mc-settings-manager:<tab>"]`: `{v, order, hidden, open}`. A section the kept order does not name goes after the section Forge put before it; a key kept for a section not on the page is kept. Storage that throws or holds nonsense is Forge's arrangement |
| Empty | An item with nothing to draw (`shows`: displayed, and a block or control inside) is not drawn (`data-mc-sm-empty`), its row greyed; a MutationObserver on the list's `class`, `style` and `hidden` attributes, a frame later, says when Gradio shows it |
| Late blocks | On every after-update: a block that appeared in the column or a container walked into is classified and joins the list before the manager |
| Off | The setting off (read live, `onOptionsChanged`): every move is undone in reverse order, the emptied containers shown, the list removed -- Forge's column as found. On again: built again with what was kept |
| A finger's drag | `touch-action: none` on the grip, and the page's scroller held `overflow: hidden` for the drag (CLAUDE.md, the Gecko trap). A drag the browser cancels (`pointercancel`, a lost capture before the release) puts the order back as the drag found it and keeps nothing |

## 3. What was found on the way

- **Gradio's button reset outranks a lone class.** `.gradio-container-4-40-0
  button { padding: 0 }` is (0,1,1): the first Reset was 58 px with its word
  spilling out of it. Every button rule here is named through its row or the
  foot (0,2,0).
- **The scratch page needed Forge's own patches to be Forge's.** Without
  `gradio.component_meta.updateable = lambda x: x` and `@wraps(gr.Checkbox.__init__)`
  on InputAccordionImpl, Gradio built the hidden checkbox from the subclass's own
  signature, with no elem_id, and Forge's `inputAccordion.js` failed on every
  accordion; with them, Hires. fix's checkbox and the link work as on Forge.
- **A real page has anchors between blocks.** Svelte keeps text nodes beside
  the blocks it draws, so putting blocks back in Forge's order "worked" on the
  real page in either order; the browser test's page has none
  (`tight()`), and there only the reverse order puts the column back.

## 4. Tests

`tests/test_settings_manager_js.py` (Chromium): the page is both columns in
Gradio 4.40's nesting with generated classes and Gradio's button reset;
`TestTheSections` (Forge's order, Img2Img's sections, a form round two
sections' controls, the list after the prompts with nothing drawn outside it,
the new accordion drawn as Gradio's, an accordion moved not copied, the chosen
script's controls in the Script section whether chosen or not, a page with no
accordion to read), `TestTheAccordions` (a label press closes and is kept;
Gradio's own never pressed or opened; the manager last and closed),
`TestTheManager` (the rows; ↑ and ↓; the eye; a mouse's and a finger's drag; a
drag the browser cancels; the arrow keys and the focus; Reset; Reset's padding;
each tab its own; hiding moves no block), `TestWhatGradioDoes` (Override Settings and Inpaint drawn when
Gradio shows them; a block drawn later; a kept place for it), `TestOffIsOff`
(the columns put back exactly, then built again with what was kept; off at
load), `TestHousekeeping` (no child list changed by later updates; storage that
throws or holds nonsense). 23 mutations, each failing a test (four survived a
first pass: the chosen script's controls, the reverse order of putting back,
the order redrawn on every press, and a form round two sections' controls
not walked into; the tests were made to see them).

`tests/test_settings_manager_live.py` runs `tests/live/settings_manager_live.py`
on real Gradio 4.40 when `MC_GRADIO_PYTHON` names a venv with it: the sections
of Forge Neo's column as Gradio really draws it, the borrowed classes and the
computed look equal to a Gradio accordion's, Gradio's own events on moved
blocks (a checkbox showing the seed's extras, the Script choice showing a
script's controls, a handler filling and showing Override Settings), Hires.
fix never opened, ↑ and the eye, and every block back where it stood with the
setting off.

## 5. Open

Not run on the user's machine, in a Gecko build or against the user's own
extensions' columns: an extension that draws a bare block in the column is
wrapped under its first label, and one whose script finds its controls through
the script container (`#<tab>_script_container …`) would no longer find them
there -- none of Forge Neo's built-in extensions or this extension's scripts
does, and none of Mini Paint NEO's. If a section looks wrong, the setting off
puts Forge's column back at once, and `mcSettingsManager.describe('txt2img')`
in the console says what was found.
