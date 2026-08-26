# The Image Pipeline — implementation notes

Companion to the *Image Pipeline UX Refactor* design intent (24 August 2026).
That document states what the refactor is; this one records the choices made
against it, the two places it was followed differently, and the handful of
things that will otherwise be rediscovered the hard way.

Section numbers below are the design intent's.


## 1. What this refactor is not

It is a presentation layer. No generation logic moved, no stage was reordered,
no failure path changed, and no execution engine was added. Every control on
the finished panel is the same Gradio component, wired to the same handler,
writing to the same state as before — in most cases the same object, moved.

The two argument lists the processing hooks receive are unchanged in length,
order and identity. `ScriptModelChain.ui()` still returns twenty controls
starting with the enable flag; `ScriptKreaCreative.ui()` still returns
`[enabled, creativity] + settings + axes + spatial`, with the Spatial tail
still last so `_split()` can find the two fixed ends without counting. That is
the property to check first if any of this is ever suspected of changing what a
press of Generate does: if those lists are intact, the generation is reading
exactly what it read before.


## 2. §2.3 — one surface built by two scripts

Creative and Spatial live in `scripts/model_chain_krea_creative.py`; Stage 2
lives in `scripts/model_chain.py`. Forge builds each `alwayson` script's `ui()`
inside a wrapper of its own, in an order this extension does not choose, so
neither script can contain the other — and "one top-level surface" is the
design intent's third non-negotiable.

`mc_pipeline_panel` resolves that without either script knowing which one it
is. Whichever calls `host()` first builds the whole shell — six rows, in
pipeline order, with two empty containers per owned stage — and remembers it
against the Blocks currently being assembled. The second gets the same object
and fills the slots the first left empty by re-entering them:

```python
with pipeline.body("creative"):
    ...the Creative editor, unchanged...
```

Gradio appends to a re-entered container and renders each container's children
in insertion order. The slots are created in pipeline order at build time, so
the panel reads Prompt → Creative → Spatial → Stage 1 → Stage 2 → Output
whichever script got there first.

**Why order-independence is the point, not a nicety.** A refactor that only
worked in one of the two orders would work until somebody installed an
extension whose name sorted between `model_chain.py` and
`model_chain_krea_creative.py`, at which point the page would grow two
half-filled pipelines and no error anywhere. `test_pipeline.py` builds both
orders and asserts one shell with all three owned stages filled, in each.

The shell is scoped to `gradio.context.Context.root_block`, which is the Blocks
being assembled right now. Forge rebuilds the whole tab on a UI reload, and a
shell remembered across that would hand the new page slots belonging to a page
that no longer exists. Where that key cannot be read the cached shell is reused
rather than rebuilt: the two ways to be wrong are not equal, and a second
pipeline on the page somebody *is* looking at is worse than a stale one on a
page they are not.

The leftover wrapper `gr.Group` the second script never fills renders as an
empty container. It is left alone deliberately — hiding it would mean a
selector against Gradio's own DOM shape, which §11's "no Gradio-generated
class" rule exists to avoid.


## 3. §2.5 — Stage 1 is read, and reading it is the whole feature

The Stage 1 row is the one place this panel could most easily have become a
liability, because the tempting version of it is a second set of size controls.
There are none. `_TRACKED_COMPONENTS` grew five entries — the checkpoint,
sampler, scheduler, steps and CFG — captured through `after_component` exactly
as the width and height sliders already were, and every one of them is an input
to a summary and never an output of one.

Every one is also optional. A Forge build that renames a control costs that
clause of the sentence and nothing else, which is the only acceptable failure
for a panel whose entire job is describing controls it does not own.

**The number that justifies the row** is the handoff. Stage 2 refines finished
Stage 1 pixels, so a Hires pass changes what it is handed, and a panel that
quoted the width and height sliders would be describing a picture that never
exists. `_stage1_size()` follows Hires; the handoff line, the Output row and
the existing Stage 2 size readout are all derived from it, so the three of them
cannot disagree about the size of one picture.

The ImageStitch reference count arrives late, because ImageStitch sorts below
Model Chain and its gallery does not exist while this panel is being built. It
uses the same one-shot deferral the reference status already used, dropped once
consumed so a rebuilt UI cannot leave two sets of handlers racing to write the
same four lines.


## 4. §5.3 — one picker, and what stayed underneath it

The design intent asks for a single multi-select Treatment field per direction,
with the count deciding the behaviour: none is no effect, one is fixed, two or
more is a seeded pool.

The implementation keeps the three machine-facing controls — mode, pinned
value, exclusions, three per axis in the library's own order — and makes them
invisible outputs of a render rather than inputs a user touches.
`apply_treatments()` is the whole translation, and it is nine lines.

This is deliberate and worth defending, because the obvious alternative is to
replace the triple with one control. That would change `axis_controls`, which
changes `_split()`, `axes_from()`, `remember_axes()`, the profile format, the
infotext contract and what `before_process` unpacks — six coupled changes to
make one control look different, against a design intent whose first
non-negotiable is that generation behaviour does not change. Keeping the triple
as a derived representation costs one render per interaction and touches none
of them. Stable ids, creativity eligibility, compatibility, anti-repetition,
user-prompt precedence and the written-expression tiers cannot tell the control
above them was replaced.

`visible=False` in Gradio removes an element from the layout while keeping its
value in the payload, which is exactly what a machine-facing control needs.

### 4.1 The state Natural could not express

An axis somebody added a moment ago and has not chosen treatments for is
Natural underneath — the Director must ignore it exactly as it ignores an axis
nobody touched — and differs only in having somewhere on screen to be filled
in. Those are two different facts and the three existing keys can only hold
one.

So `krea_creative_directions` records which axes have a row. It is a fact about
the panel, kept by the panel, and the generation never learns a state it would
then have to ignore. `known_directions()` treats it as additive: any axis that
is actually directing has a row whether or not the key mentions it, so a
settings file or a profile written before the picker existed still shows its
directions.

Profiles carry it, including the two computed built-ins — which turned out to
be building their values without going through `normalise()`, so adding a field
to `FIELDS` broke `apply()` for both until they were fixed. Worth knowing about
that pair: they are the only profiles whose contents are code rather than data.

### 4.2 Vary with no exclusions is every treatment selected

That is what it has always meant — "the director may choose anything" — now
said in the vocabulary of the picker. It matters because it is what makes the
mapping round-trip: `selection()` and `apply_treatments()` are one mapping read
in two directions, and halves that disagree are a panel that forgets a choice
the moment the tab is reloaded.


## 5. §6 — two canvases, one document

The compact canvas has one verb. It moves the box under the pointer, and it
cannot create, delete, rename, resize, restyle, restack or reframe anything.

The list of what it cannot do is as load-bearing as the one thing it can. A
compact canvas that could also delete a region would be a second layout system
competing with the first over the same document; the restraint is what keeps it
a shortcut into that document. `test_krea_spatial_js.py` asserts the absence
against the markup, because a control that is not in the page cannot be reached
by an interaction a test forgot to try.

**Hit testing is the browser's.** Regions are elements appended in paint order,
so `event.target.closest()` returns the topmost one under the pointer by
construction — no z-order arithmetic, and no way for the answer to disagree
with what somebody can see.

The two canvases share the file, the serialization and the document, and share
nothing else: the compact canvas has its own working copy, its own undo stack
and its own dirty flag, because the full editor's `state.working` is null while
the editor is shut and the two are never open at once. `serializeIn()` and
`orderedIn()` were factored out of their `state.working`-bound originals for
this; two orderings of one layout is how a box you can see behind another
becomes the one that moves.

**One trap, found by an existing test.** `once()` guards its listeners with a
dataset flag, and both canvases follow the txt2img size fields. Sharing the
flag name meant whichever wired first silently took the other's listener away —
which showed up as the *editor's* frame no longer following a size change. It
now takes a tag.


## 6. §8 — loaded, modified, not saved

`mc_profile_state` gives the three features one vocabulary. It formats a string
and compares two snapshots; it writes nothing and owns no storage.

The rule that needed a shared home is §8.3, because the two readings of "not
saved" are one word apart in English and opposite in consequence:

```
not saved  ==  the named profile still holds its old values
not saved  !=  your changes will be ignored
```

So the sentence saying which one it is appears on the panel beside the state,
not in documentation. The moment somebody needs it is the moment they are
looking at the word "unsaved" and deciding whether to trust the screen.

Two implementations, for two different shapes of problem:

* **Creative and Spatial recompute.** Every handler on the Creative panel ends
  in a full render, and the Spatial document is written by a browser file the
  server does not hear from between presses — so "does the screen still match
  what it came from" is answered by comparing the two. A derived flag cannot
  drift out of step with the thing it describes.
* **Stage 2 tracks a baseline.** Its nineteen controls have no single funnel,
  so applying a preset records a fingerprint in a `gr.State` and every control
  reports against it. `change` rather than `input`, deliberately: `change` also
  fires when the server writes a value, which is what makes applying a preset
  settle back to clean instead of reporting the load as an edit.

Snapshots are JSON rather than tuples or hashes because Gradio's transport
turns tuples into lists, and a dirty flag that reported a round trip as an edit
would simply always be on.

### 6.1 §8.5 — the one place it has to be said twice

Spatial has two saves that are genuinely different. Auto Save commits the
*working layout*, which is what the next Generate composes. Save updates a
*named layout*, which is a copy somebody asked for by name. `Loaded: Studio
thirds · Modified · not saved` is therefore an ordinary state to sit in for as
long as you like, and dragging a box never silently rewrites a layout you
named.

`mc_spatial_profiles.matches()` compares parsed documents rather than strings,
and drops the canvas size before comparing: the editor rewrites key order and
whitespace every time it serializes, and the frame is a fact about txt2img
rather than part of the composition. Comparing bytes would report a layout as
modified for having been opened, and every saved layout as modified for a
change of generation size.


## 7. §4 — the animation, and why it reads the progress bar

`mc_progress` already runs one ordered phase list for the whole job and writes
the current phase's label into the progress response, which the host prefixes
to the bar's text. The pipeline animation reads that label and lights the
matching row. It has no timer, no percentage arithmetic and no state of its
own — §4 asks for cosmetic system status, and the way to keep it cosmetic is to
give it nothing to be wrong about.

A skipped stage is never entered and so never lights, which satisfies "skipped
stages receive no active animation" without the browser needing to know which
switches are on.

Matching labels as text is a real coupling between a Python constant and a
JavaScript table, so it is tested rather than hoped for. `test_pipeline.py`
parses the table out of the JS file, re-implements the browser's
first-substring-wins rule, and drives it from a real `mc_progress.build()`
plan — so a phase added on one side without a rule on the other fails, and so
does a rename on either.

`WAITING` is said by both language-model passes and therefore names no stage on
its own. It is treated as ambiguous and keeps whatever was already running: the
writer and the composer each announce themselves on the phase either side of
their wait, so "still that one" is always right after the first.


## 8. What a first week of using it changed

Four things, all of them from the same complaint: the panel was spending screen
space on sentences rather than on settings.

### 8.1 Three of the six rows are gone

Prompt, Stage 1 and Output were drawn muted and uneditable so that the path had
no holes in it (§2.4). What three holeless rows produced in practice was a
restatement of what the page already said louder — the prompt box is directly
above the panel, the size sliders directly below it, and the output is the
picture. The panel is the three stages this extension actually runs.

One number outlived them, because nothing else on the page states it: the pixel
size that crosses into Stage 2, which a Hires pass changes (§3 above is still
the reason). It is drawn on the edge above the Stage 2 row and is still true
when Stage 2 is off.

Section 3.3's note outlived the Prompt row too, and is now the whole of what
that element says: `2 literals active`, when the Literal Prompt boxes are off
screen and carrying something. Empty otherwise, and an empty note has no height.

Stage 1 and Output are still *phases*, and `model_chain_pipeline.js` still
recognises every label the progress bar produces for them — it has no row to
light, which is a different thing from not knowing the label.
`PHASES_WITHOUT_A_ROW` is what keeps the tests able to tell those two apart.

### 8.2 Each row is a surface

On a dark theme the rail and the text were the only two things on the panel, and
three stages of unboxed text read as one paragraph with bold words in it. Each
stage has a background and an edge now, and the rail is drawn in the gaps
between them so it still says these things happen in order.

### 8.3 The descriptions went behind an "i"

    Spatial Layout: 7 regions. Region prompts are used exactly as typed.
    Direct BBOX Merge: your prompt is used exactly as typed as the global scene
    and your regions are applied deterministically. No language-model request is
    made — the fastest and most predictable Spatial option.

One number in that. The rest describes a mode that has not changed since it was
chosen, repeated on every render. `mc_hint` builds a badge — a span with the
text in an attribute, a pseudo-element for the bubble, no JavaScript and no
popup library, with the browser's own `title` as the fallback — and the rule it
applies is: **live data stays on the panel, description goes behind the "i"**.

Paragraphs already inside collapsed drawers were left alone. They cost nothing
until opened, which is the same bargain.

### 8.4 The two panels opened on the wrong control

Stage 2 opened on its checkpoint chooser with Presets in the last of six
accordions; Spatial opened on four controls about storage above the canvas they
are storage for. Both had the decisions in the wrong order — a Stage 2 preset
*is* the checkpoint, and a saved layout is loaded once while the canvas is
worked in continuously. Presets are the top of Stage 2 now, with the checkpoint
in a section beside the modules and residency status that describe the same
model; the canvas is the top of Spatial, with saved layouts in a drawer above
Spatial options.


## 9. Where the design intent was followed differently

**§3.4, mutual collapse.** "Opening another owned stage *may* collapse the
previously open stage." It does not. Closing another `gr.Accordion` from
JavaScript means reaching for Gradio's own generated classes to find its
disclosure control, which is the one thing every selector in this extension
avoids — a theme is allowed to rearrange Gradio's internals, and the panel has
to keep working when it does. The word in the design intent is *may*, and the
compactness it buys is not worth a selector that a theme update can break.

**§9.8, diagnostics.** The Generation Memory section stays outside the Image
Pipeline, where it already was. What it describes is the memory policy for
*this generation*, which exists whether or not Stage 2 is armed — a plain Stage
1 press has a plan, a peak and a persistent LLM allowance just as a long chain
does — so a section that only appeared alongside Stage 2 would suggest the
policy did too. It is not one of the three surfaces §2.3 names.

**§6.5, the full editor's placement.** The BBOX workspace is built at the top
level rather than inside the Spatial stage's drawer. §6.5 and §10 both give it
a dedicated large area, and a workspace nested three containers deep inside an
accordion is one `overflow: hidden` or one `transform` away from being a window
nobody can see — which is the exact bug the workspace was moved out of an
overlay to fix. It is hidden until Full Screen reveals it, so it costs no space
in the pipeline and competes with nothing.


## 10. The stage card, and everything that starts closed (26 August 2026)

The panel worked and read badly. Each stage was four things stacked — a title
row, a switch, a summary line, and *then* an accordion labelled with a
restatement of the stage's own name that had to be opened to reach the
settings. Two rows saying "Creative", one above the other, and only the lower
one opening anything. Inside it, Creative's twenty axis rows unfolded the
moment the stage did.

### 10.1 One disclosure, and the switch is not on it

§1.3 of the redesign intent asks for one large disclosure surface, one
enable/bypass switch, one live summary, one attached body, and no redundant
second "open me" row. The card is now the stage's own accordion, labelled with
the stage's name, and that is the only disclosure.

The switch and the summary cannot be built into it. Gradio gives an Accordion a
plain string for a label — no slot, no components — so a header carrying a live
line and a checkbox is not something Python can express. They are built as
siblings of the accordion and moved into its header by
`javascript/model_chain_pipeline.js`.

That file finds the header without naming a single Gradio class. An Accordion's
disclosure is its first element child and it is the thing that carries the
click handler, so it is recognised by what it *is* — a button, or an element
that says whether it is expanded — rather than by what a Svelte component
happened to call it. Deliberately not a `querySelector` for a button anywhere
inside: the first button inside an accordion that has no header button is a
button in its *body*, and furnishing that would move the enable switch into the
settings.

Two classes go on when it works: one on the header, one on the card. Every rule
in the stylesheet is keyed on those, so if the browser file never runs, or a
Gradio version renders a header it cannot recognise, nothing is moved, no class
is added, no rule matches, and the card falls back to name / summary / switch
stacked in ordinary flow. Every control still works and only the arrangement is
lost, which is the right thing to lose.

A click on the switch is stopped at the switch rather than the header being
taught about exceptions. Arming a stage never also opens it.

### 10.2 "Bypassed", not "Off"

§1 again: a bypassed card stays visible and configurable, so its summary is the
only thing standing between "this setting exists" and "this setting will run".
Every stage that is switched off now begins its summary with the same word.

The summaries themselves did not need changing in any deeper way, and that is
worth recording rather than assuming: they were already what §4 of the intent
asks for. `_creative_line`, `_spatial_line` and `_stage2_summary` are selectors
over the stored settings — `summarize(canonical)` and never
`summary = what_the_user_just_clicked` — so a collapsed card cannot disagree
with the expanded controls behind it. What changed is one word in three
functions.

### 10.3 Everything starts closed, and what is opened is remembered

`mc_pipeline_panel.drawer()` is now the only way this extension makes a
disclosure on txt2img. It has no `open` parameter — not a default a caller may
override, an absence — and it stamps one class on everything it builds.

One class, three jobs: the stylesheet gives every drawer the same outline and
the same header, the browser file remembers which ones were opened, and a test
counts them. A feature that built a bare `gr.Accordion` would get none of that,
which is why there is a test asserting that no file drawing on txt2img contains
one.

Closed is the right answer exactly once. After that, a tab that folds everything
away again on every reload is a tab somebody has to re-open four drawers in
before they can carry on — so the open ones are remembered by element id in
`localStorage`. Furniture, stored where furniture belongs: nothing about a
generation is kept there and nothing there is ever sent anywhere. Every read and
every write is inside a `try`, because private browsing, a blocked store and a
value somebody edited by hand all mean the same thing — no preferences, which is
what this shipped with.

Restoring is a *press*, not a class. Gradio owns whether an accordion is open
and re-renders it from its own state, so setting the class would leave the two
disagreeing at the first update. Pressing the header is what the user would have
done.

### 10.4 Creative is four drawers at one level

§1's Creative hierarchy change, and the first clause of the final acceptance
statement. Profile and Creativity are top-level; Create a profile, Directions,
Advanced settings and Recovery & diagnostics are same-level siblings, and
nesting begins only inside one of them.

Directions used to *be* the body of the panel with everything else arranged
around it. The first screen of the stage was a list of axes nobody had asked
about yet. It is a drawer now, and its label carries the count — `Directions —
2 active` — which is a derived selector like every other summary here and is
returned by `render()` with everything else, so it cannot disagree with the rows
inside it.

The Save As name box moved with it. It used to be a row that appeared out of
nowhere when a button was pressed; it is the contents of "Create a profile" now,
which is a thing somebody can find without pressing Save As first to discover
what Save As does. Save As opens that drawer, and nothing ever closes it — a
render that folded it away would do so while somebody was typing into it, on any
handler that happened to fire.

The Creativity slider is built by `build()` rather than handed to it, because
where it goes is part of the shape: §3 puts it beside Profile at the top level,
and a caller that made it first would put it above them both. LLM Studio's Krea
tab is the exception and passes its own, because there the slider lives outside
the panel entirely and shows and hides with Creative Mode.

### 10.5 Structure in geometry, not in palette

§2's objective is that the extension look native inside somebody's Gradio theme
without losing the hierarchy. So the palette is the theme's decision and the
geometry is the stylesheet's.

Every colour, radius and shadow is a semantic token mapped to a host variable
with a fallback — `--mc-pipe-border` to `--block-border-color`, `--mc-pipe-accent`
to `--color-accent`, and so on. Nothing below them is a literal colour. What
carries the structure instead is the list §2 calls invariant: a closed outline
on each card, a separator between a header and its body, a nesting indent, a
one-pixel rail down the left of nested content, an outline on every drawer, an
outline on every direction row, and read-only context shaped like nothing that
could be edited.

A theme whose block fill and page fill are the same colour still gets a
structured panel, because none of the structure is carried by a fill.

Touch targets are sized in `em` so they follow the theme's type scale, and a
coarse-pointer media query takes them to a literal 44px.

### 10.6 What this refactor did not do

Pages 4 and 5 of the intent describe a canonical `PipelineSettings` structure
with a monotonic revision, expected-revision checks on every mutation, a
pending/acknowledged state per control, and a Generate that blocks until
`ui_ack_revision == model_revision`. None of that is here, deliberately: it is
an engine change and this was asked for as a surface one.

It is worth saying which half of it is already true. The settings *are*
canonical and server-side — `mc_creative_krea`, `mc_spatial` and the preset
store are the one authoritative representation, every handler writes through
them before returning anything, and every summary on the panel is derived from
them rather than from a control's value. What is missing is the revision guard:
there is no version number on a mutation, and nothing stops a user moving a
slider and pressing Generate before the callback lands. That is a real race and
it is unaddressed.
## 11. What it looked like on the theme most people use (26 August 2026)

Section 10 shipped and looked right under Lobe and wrong under stock Gradio:
every stage card painted white with light text on it, the switch and the summary
stacked underneath the name instead of beside and below it, and a checkbox for a
touch target. Two separate mistakes, and the second one hid behind the first.

### 11.1 A fill and a text colour are two guesses that have to agree

The token adapter gave the cards `--panel-background-fill`, with
`--background-fill-secondary` behind it, and took their text from whatever the
host was already using. On stock Gradio those two came out light-on-light.

The fix is not a better variable. It is that **the panel paints nothing**. Cards,
drawers, bodies and direction rows are outlines over whatever the host is
already painting, they inherit its text colour, and there is no pair left to
disagree. Every fill variable — `--panel-background-fill`,
`--block-background-fill`, both `--background-fill-*`, `--body-background-fill`,
`--input-background-fill` — is now absent from the block, and a test says so by
name because the way this regresses is somebody adding one back to make an edge
show up.

What still paints a background is the rail, its elbow and the stage node: one or
two pixels each, in a border colour, with no text on top of them.

This is what §2 of the intent means by "use colour as a secondary signal only",
read strictly. Structure that survives an arbitrary theme is structure carried by
geometry, and the strongest form of that is a panel with no palette of its own at
all.

### 11.2 The header move worked under one theme and found nothing under the other

§10.1 built the switch and the summary beside the accordion and had the browser
file move them into its header, recognising that header as "a button, or an
element that says whether it is expanded". Lobe rebuilds the header as a
`<button>`, so it worked there. Stock Gradio renders a `<span>`, so it found
nothing, took the documented fallback, and the fallback is what shipped: three
pieces of a header stacked down the left of the card.

The fallback was correct and the design was wrong. Nothing needs to move.

A Gradio Accordion is two elements — the thing you press and the thing it shows —
so `.mc-pipeline-editor > :first-child` **is** the header, under every theme and
every version, and it names no class of Gradio's. The stylesheet reserves a band
in it: a name line, a summary line under that, and a lane on the right. The
summary and the switch stay exactly where Gradio put them and are painted into
that band with `position: absolute`, which is also what keeps them in the header
when the card is open — as ordinary siblings they would be pushed below
everything the body contains.

So there is no DOM surgery left in the panel at all. The browser file's only
remaining interest in a header is pressing one to restore a drawer somebody had
open, and even that now finds it by counting children rather than by guessing at
a tag.

### 11.3 Targets

The enable switch was a bare 13px checkbox in the corner of a card, and it is the
one control a collapsed pipeline has. It is a pill now: outlined, at least 4.4em
wide and one tap tall, with a 1.25em box inside it, and both the outline and the
tick say whether the stage is armed — §2 again, never state by colour alone.

Everything in a stage body — buttons, text and number fields, selects — has a
minimum height in `rem`, and a coarse pointer takes the lot to a literal 44px.

The composition mode is the two large segmented targets §3 asks for rather than
two radio dots. It is still the same stock `gr.Radio`: same component, same
handler, same value, wearing the shape.

### 11.4 Two presses before something is gone

§3 asks for an explicit confirmation where the loss is irreversible. Deleting a
Creative profile, a Stage 2 preset or a named Spatial layout removes a file, and
nothing brings it back; all three used to go on the first press.

The confirmation is the button. The first press arms it, it changes to *Confirm
delete*, and the status line says which thing is about to go; the second press
does it. A modal would be a second thing to dismiss on a tab that has enough of
them, and a browser `confirm()` is not styleable, not themeable and not
touch-friendly.

The armed flag is a `gr.State` and not a module variable, because an arm is one
person's half-finished gesture in one browser and a flag on this process would be
shared by every tab open on the server. Nothing else clears it, deliberately: an
arm that expired on the next unrelated click would be a confirmation somebody
could miss by being slow, and the only cost of it lingering is a button that has
to be read.

One guard, `mc_pipeline_panel.confirmed()`, in all three places — and a test that
fails if a fourth delete appears without it.
## 12. One omitted line broke both controls (26 August 2026)

Section 11 replaced the header move with a band the switch is painted into, and
shipped with the switch spanning the whole card: an orange pill wrapped around
the stage's name and its summary, and tapping anywhere on the header armed the
stage instead of opening it.

### 12.1 An absolute box with no width is not shrink-wrapped

`.mc-pipeline-stage > .mc-pipeline-switch` set `position`, `top`, `right` and
`z-index`, and no `width`. An absolutely positioned box with `width: auto` does
shrink-wrap -- but only if nothing else sets a width, and Gradio gives every
block `width: 100%`. That rule won by default because this one never made a
claim, so the switch box became the whole card anchored to the right edge, and
the label inside it filled the box.

A `<label>` toggles its checkbox wherever it is pressed. So the label was the
header: the name, the summary and the empty space all armed the stage, and the
accordion underneath could not be reached at all. Two controls that were meant
to share a line, and one omitted declaration took both of them out -- the switch
by making it enormous, the disclosure by burying it.

The box is bounded twice now: `width: auto` is what makes it shrink-wrap, and
`max-width: calc(var(--mc-pipe-lane) - 0.8em)` is what stops a long label or a
theme with generous padding creeping back over the name. The subtraction keeps
it clear of the chevron, which Gradio puts at the inner edge of the padding the
lane is made of.

That the switch is a *sibling* of the accordion rather than a child of its
header is what makes the arrangement work at all: its presses never reach the
accordion's own click handler, so there is nothing to stop propagating and
nothing to get wrong. The summary over the rest of the band carries
`pointer-events: none`, so presses there fall through to the disclosure.

### 12.2 The pill is gone

It was drawn to make the switch look like a control. In the header of a card
that already has an outline it reads as a second card edge, and in the accent
colour it reads as an error. A checkbox is already legible as a control, and
ticked-or-not is already a state signal that is not a colour -- which is all §2
asks for. What is left is the hit area, which is the part that had to be big:
one tap tall, with a 1.3em box in it, and no box around any of it.

### 12.3 Nothing here is worth a spinner

Every handler on this panel repaints text somebody is looking at -- a summary
line, a status note, a count. Gradio's default draws a progress overlay over
each output for the length of the round trip and takes it away again, and on a
panel where one press changes three lines that reads as the card blinking.

All 48 of them now pass `show_progress=False`, and there is a test that walks
every `.click`, `.change`, `.input` and `.release` in the three files that draw
on txt2img and fails on one that does not. They were all `queue=False` already
-- no work worth queueing, nothing that starts, stops or waits for a generation
-- and a handler that is not worth queueing is not worth animating either.

The other half of the flicker was layout. The summary is painted into a reserved
band, so its text can change length or empty entirely without the card resizing.
And the Literal Prompt row, which sits in the middle of the prompt area, is no
longer re-sent its own visibility on every toggle: the row is visible when
either stage is on, so flipping one changes the answer only when the other is
off, and that is derivable from the two values the handler already receives.
When it is not the deciding vote it now says nothing at all, and the prompt area
does not reflow so that nothing can change.
## 13. Three rows, and nothing else (26 August 2026)

Under Lobe the panel showed a stray ✕ between Spatial and Stage 2, each stage's
description sitting outside its card rather than under its name, two node
markers down the left where there should have been none, and a switch whose
invisible box overlapped the chevron so the switch could not be tapped.

Four symptoms, and reading them together says the same thing four times: the
card had grown parts that were not the card.

### 13.1 What a stage row is

A name, a one-line description, a switch, and a way in. That list is exhaustive,
and everything outside it went:

* **The title row above the disclosure** -- gone in §10, and this is where the
  last of it goes: the name is the disclosure's own label.
* **The rail and the node beside every card.** A vertical timeline is a good
  idea in a mockup and a duplicate in a themed page: Lobe draws its own bullet
  on every accordion header, so each stage had two markers half a line apart.
  The order of the rows already says the order they run in.
* **The handoff line between Spatial and Stage 2.** A pipeline of three stages
  should be three rows, and a fourth thing between two of them that is not a
  stage, cannot be opened and cannot be switched off is furniture. Under Lobe
  it rendered as a stray glyph, which is what furniture does when nobody owns
  its styling. The number is Stage 2's -- it is what Stage 2 is handed -- so it
  leads Stage 2's description: `1536 × 2304 in · Klein 9B · denoise .35`, and
  `1536 × 2304 in · Bypassed — the Stage 1 image is the final image` when it is
  not armed, because that is exactly when somebody is deciding whether to arm
  it.

### 13.2 The description belongs in the label

Three arrangements were tried for it and each failed on a theme:

1. Build it beside the accordion and move it into the header from JavaScript.
   Worked under Lobe, where the header is a `<button>`; found nothing under
   stock Gradio, where it is not.
2. Paint it into a reserved band with `position: absolute`. Worked under stock
   Gradio; fell outside the card under Lobe, whose header is a different height
   than the one the offsets were measured against.
3. Leave it in flow after the accordion. Puts it under the *body* the moment the
   card is open.

The fourth has none of those failure modes because it is not a second element.
A Gradio Accordion's label is a string, and a string with a newline in it is two
lines: `Creative\nC7 · 2 directions · Editorial`. `white-space: pre-line` shows
the break and `::first-line` makes the name a name. Whatever a theme does to the
header, the text is in it -- and a theme that refuses `pre-line` gets one line
reading "Creative C7 · 2 directions", which is a worse layout and still the right
information in the right place.

So `pipeline.summaries[stage]` **is** the accordion now, and a feature repaints
its description with `card_summary(stage, text)` -- an update that sets a label
rather than replacing a component. That is also one less thing that flickers:
setting a label replaces text, where replacing a Markdown beside it tears an
element out of the page and builds another one.

### 13.3 The chevron had to become ours

The switch is painted into a lane the header's padding reserves. That works only
if nothing else lands in the lane, and the one thing on the right this file could
not place was Gradio's own chevron: under stock Gradio it honours the header's
padding, under Lobe it is pinned to the edge. Either way it and the switch
eventually meet, and the report was the plain consequence -- the switch's hit
area under the chevron, and neither reachable where they overlapped.

Gradio's chevron is hidden -- it is the header's last child, which is structure
rather than a class name -- and replaced by a caret this file draws at a position
this file chose. The header now reserves two zones on its right, the lane and the
caret, and both of them are ours. Nothing can arrive between them.

### 13.4 What is left of the running indicator

The pulse and the filled node went with the rail. A running card takes the accent
on its own border and its own name, which is the same information with nothing
new drawn on the page -- and one fewer animation on a panel whose last round of
feedback was about flicker.

## 14. Two lines, and what it took to keep them two (26 August 2026)

> "now the title and description live on the same line. please make this two
> lines for each pipeline step. the description on second line should be styled
> differently than title, and truncate if too long."

§13.2 put both lines in the accordion's own label, which is the only place a
Gradio disclosure can carry text that is guaranteed to be inside its header. The
text went in. The *break* did not survive, and neither did the difference between
the two lines. This round is four separate reasons why, each found by rendering
the real stylesheet in a browser rather than by reading it.

### 14.1 The card header was being styled as a generic drawer header

`.mc-pipeline-editor > :first-child` and `.mc-pipeline-drawer > :first-child` are
both `(0,2,0)`, and the stage card's accordion carries both classes. Equal
specificity, so the one further down the file won -- and that was the generic
drawer rule, which sets `padding: 0.4em 0.7em`, `font-weight: 700` and
`text-transform: uppercase`.

Every visible symptom followed from that one line. The header had no lane
reserved on its right, so the description ran underneath the switch. It was bold
and uppercase, so both lines looked like a title. The generic rule is for the
drawers *inside* a stage body, and it now says so:
`.mc-pipeline-drawer:not(.mc-pipeline-editor) > :first-child`.

### 14.2 A shared measurement cannot be in `em`

The header and the switch are two elements at two font sizes -- the header is
`var(--text-sm)`, the switch is whatever the theme gives a checkbox -- and
`--mc-pipe-head`, `--mc-pipe-lane` and `--mc-pipe-caret` are a bargain *between*
them: the header keeps its text out of the lane, the switch sits in it. In `em`
each element resolved the same number against its own font size and got a
different answer. Measured: a band `3.9em` tall came out 54.9px at the header and
62.4px at the switch, so the switch was taller than the band it is painted into
and reserved a wider lane than the header kept clear.

Those three are `rem` now. The two that are genuinely per-element -- the indent
and the gap -- are still `em`.

### 14.3 `inherit`, and never the value itself

Gradio wraps the label text in a span, and a theme may style that span. Lobe
gives it `nowrap` and a weight of its own, which beat what the header passed
down: the break disappeared and the two lines came out as one bold run sliding
under the switch.

The obvious repair is to restate the header's font on the span. It wins the
specificity fight and loses the point -- a declaration on the span *also* beats
what `::first-line` hands down, so the name would come out the same size as the
description and there would be nothing to tell the two lines apart. What the span
needs is not a value but an absence of opinion:

```css
.mc-pipeline-editor > :first-child * {
    white-space: inherit !important;
    font: inherit !important;
    color: inherit !important;
}
```

Now the header sets the second line, `::first-line` sets the first, and the span
carries whichever reaches it. Measured in Chromium against a stylesheet that
styles the span the way Lobe does: values on the span give one line at one size,
`inherit` gives two lines at two.

### 14.4 `pre`, not `pre-line`

Both honour the break in the label. Only `pre` refuses to add one of its own.
Under `pre-line` a description a little wider than the column soft-wrapped to a
third line, which the fixed band then clipped mid-word -- and because the line had
wrapped, `text-overflow` had nothing to do, so there was not even an ellipsis to
show that anything had been cut. `pre` pins the header at two lines whatever the
column is doing, and makes the ellipsis mean something.

### 14.5 The cut is made twice

`mc_pipeline_panel.SAID` is 30 characters, and the header ellipsises whatever
still does not fit. Two cuts because they answer different questions: the CSS one
knows the column width and the Python one survives a theme that flattens the
break. The number is measured -- about 27 characters fit across Forge Neo's
generation column at the size the second line is set in -- and the three
descriptions a fresh install shows are all inside it, because those are the ones
somebody reads before they have touched anything:

| stage | bypassed description |
| --- | --- |
| Creative | `Bypassed — prompt as-is` |
| Spatial | `Bypassed — 4 regions` |
| Stage 2 | `1024 × 1024 in · Bypassed` |

Stage 2 keeps the handoff size when it has one and drops the explanation instead,
because the size is the number somebody reads in order to decide whether to arm
the stage. Without a size there is room to spell it out: `Bypassed — Stage 1 is
final`.
