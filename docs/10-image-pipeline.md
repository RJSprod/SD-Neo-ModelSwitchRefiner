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


## 8. Where the design intent was followed differently

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
overlay to fix. It is hidden until Edit Layout reveals it, so it costs no space
in the pipeline and competes with nothing.
