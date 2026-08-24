# Literal Prompts — implementation notes

Companion to the *Literal Prompts UX* design intent (24 August 2026). That
document states what the feature is; this one records the choices made against
it, the two places it was followed differently, and the handful of things that
will otherwise be rediscovered the hard way.

Section numbers below are the design intent's.


## 1. What this is, in one sentence

Two prompt boxes whose contents become `LiteralCommand` objects.

That sentence is the whole architecture, and everything below is a consequence
of taking it literally. There is no field parser, no field assembly step, no
generated bracket string and no second prompt path — by the time anything
downstream sees a field's text it is a command in the sidecar
`literals.parse()` already produced, indistinguishable from a `[[…]]` somebody
typed, restored by the same single call to `literals.restore()`.

`tests/test_krea_literals.py` asserts the indistinguishability directly: the
same string, once through a bracket and once through a field, produces commands
with equal payload and equal placement. If those two ever diverge, one of them
has acquired an opinion about LoRA tags.


## 2. §4 — priority is a tuple, not a sort

`literals.merge()` builds one tuple:

```
(*explicit prefixes, field prefix, field suffix, *explicit suffixes)
```

`split()` reads placement groups in tuple order, so putting the commands in
that order *is* the priority. Nothing sorts, nothing compares, and
`source_order` is untouched — it records where a command was in the text it came
from, and a field did not come from any text.

The rule it encodes is that typed syntax outranks a field **outwards**: an
explicit prefix sits further from the body than the field prefix, an explicit
suffix further than the field suffix. That is what somebody is asking for by
reaching for the syntax at all — positional control the fields do not offer —
and it is the only arrangement in which adding a field to an existing prompt
cannot move something already placed by hand.

`merge()` returns its argument unchanged when both fields are empty, which is
the case it is called in almost every time. Identity and not equality: the parse
handed in has to come back out untouched, because the no-brackets path is the
one that has to stay byte-identical for llama.cpp's prompt cache.

### 2.1 One field is one payload

Never split on commas, newlines or anything else. Deciding where to cut a field
would be interpreting it, and the order of the pieces would then be this
extension's opinion rather than the user's. `<lora:a:1>, __b__, $c` is one
command carrying three syntaxes belonging to three owners, exactly as the same
text inside `[[…]]` has always been.


## 3. §12 — where the two values live in the argument tuple

They go in the **variable middle**, immediately before the Spatial tail. That
looks backwards and is the only placement that works.

`mc_plan.creative_from()` reads the Spatial controls off the *end* of
`script_args` — `args[-SPATIAL_TAIL:]` — because the axis block in the middle is
a variable length and the two ends are the two that can be read without
counting. Appending the literal controls would have moved the end out from under
it and described every Spatial generation's plan wrongly: the wrong phases
reserved, the wrong VRAM arithmetic, and a progress bar naming a pass that was
not going to run.

`tests/test_plan.py::TestTheArgumentIndicesAreRight` caught it, which is worth
recording because it is the second time in two refactors that this test has
caught an argument-tuple change made three files away.

`_split()` therefore recognises three shapes rather than two, and the middle one
matters: a caller sending the tuple as it was before these boxes existed is cut
in exactly the places it always was and contributes no fields, with the saved
values answering instead. `panel_values()` in the literal tests can build that
older shape on purpose, so the compatibility is exercised rather than asserted.


## 4. §3.3 — hidden, persisted, and therefore announced

The two values persist as preferences, for the reason the Spatial canvas does:
a filter LoRA somebody always wants is configured once, and a restart that
quietly emptied it would change what the next generation produces without
saying so.

That cuts both ways. A value still in effect while its row is off screen is
exactly the invisible active state this extension keeps warning itself about, so
the Prompt row of the Image Pipeline says `2 literals active` whenever the row
is hidden and the boxes are not empty. The announcement is the price of the
persistence, not a nicety on top of it — and it is why
`test_the_panel_has_no_hidden_plumbing_for_the_browser` accepts the row by name.

The merge happens **before** the "neither feature is on" check in
`before_process`. A field that only worked while Creative or Spatial was running
would be a field whose row is hidden exactly when it stops working, which is the
opposite of what §3.3 asks for. Protection here is about delivery, not about
anything having been protected from.


## 5. §10 — the paste empties the boxes

This is the one place the design intent offers a choice and reproduction takes
it away. §10 says workflow restoration *may* reconstruct the friendly fields,
and also says the new UI must not break exact reproduction of older
generations. Those two pull opposite ways, and the second one wins.

A pasted image's recorded `Prompt:` already contains these payloads — they were
restored into it before Forge wrote the infotext. Refilling the boxes as well
would insert them a second time, and the picture would not reproduce. So a paste
empties them, by the same mechanism and for the same reason that it switches
Creative Mode off.

Keyed off `RESTORED_BY_PASTE` — any of this extension's prompt-transforming
records — rather than off the literal keys alone. One of our images made with
empty boxes writes no literal key at all, and leaving somebody's current boxes
in place would add text that image never had. An image carrying none of those
keys returns `None` and the boxes are untouched: an ordinary PNG should not be
able to empty a control any more than it can switch a feature off.

Nothing is lost. The record is stashed on `mc_creative_krea.pasted`, the
Creative drawer prints what the boxes held, and **Restore Creative setup** puts
them back — restored rather than merely shown, which is the difference between
these and the `loras` field an older image records: those controls still exist.

### 5.1 present, recorded, literals

`Setup` grew two properties that are deliberately kept apart:

* **`present`** — is this a Creative image? Decides whether a paste says so and
  whether restoring turns the writer back on.
* **`recorded`** — is there anything here at all? Decides whether the record is
  kept by `Pasted.remember`.
* **`literals`** — did it record the two fields?

Two Literal Prompt boxes with both features switched off are a whole restorable
setup and not a Creative one. Folding them into `present` would have printed
"Creative image restored" over an image whose writer never ran; leaving them out
of `recorded` — which is what `remember` originally tested — silently dropped
the record before anything could show it.


## 6. §6 — region literals reuse the same merge

`spatial.py` calls `literals.merge()` with `scope=REGION` and the region's id.
One merge, one ordering rule, two scopes; the region's `describe()` already put
prefixes before the words and suffixes after the hints, so nothing about
assembly changed.

The two raw field values are kept on `Region` beside the merged payloads, for
the reason `raw_prompt` is kept beside `prompt`: a round trip that returned only
the payloads could not tell a field from a `[[…]]` typed in the prompt box, so
reopening the editor would move a command out of one and into the other,
silently, once. Both keys are written only when they carry something, so a
layout drawn before these fields existed serializes to the bytes it always did.

The browser stores two strings and merges nothing. Every decision about what a
payload is, which side it goes and how it combines with the region prompt
belongs to Python; a browser with a vote would be the second parser §12 forbids.
A field containing `+[[not parsed here]]` is stored verbatim, and that is the
test for it.

The compact canvas gains nothing, asserted against its markup rather than by
trying every interaction a test might forget to try.


## 7. §2 and §7 — moving somebody else's DOM

Forge gives an extension no way to build a component into the prompt area:
`ui()` runs inside the script accordion, several hundred pixels below the box
these two belong beside. So they are built as ordinary Gradio Textboxes — real
components, real handlers, real values in the payload — and
`javascript/model_chain_literals.js` moves the finished row with one
`insertBefore` of an element Gradio addresses by id.

That is acceptable only because nothing depends on it having worked. If every
line of that file fails, the row is still on the page one scroll further down,
with two working boxes whose values still travel with every press, and the LoRA
browser still behaves exactly as it did before. `TestItCannotAffectAGeneration`
holds the file to that: no Generate button, no timer, no writing into a prompt
box, and no knowledge of any payload type.

The prompt family is the host's `activePromptTextarea`, joined on focus exactly
as Forge joins its own two. There is no formatter here, no caret arithmetic and
no insertion code — all three already exist, are already correct, and are
already what the positive and negative prompt get. The half of §7 about not
resetting the target is satisfied by doing nothing: the Extra Networks browser
is not a prompt box, so opening it fires no focus event and the remembered
target survives.

**§8, Tag Autocomplete and LoRA autocomplete.** They carry the host's own
`prompt` class, which is the pattern that extension matches on
(`.prompt > label > textarea`, in its `_textAreas.js`), and they are stock
Textboxes with nothing custom about their text entry. That was written here as a
well-founded expectation; §10.2 below is what happened when it was checked.


## 8. §9 — the CFG collapse touches nothing

A class on Forge's own component, taken off again the moment CFG rises. The
value is never read, never written and never cleared, and the test that matters
most redefines the textarea's `value` property and asserts the collapse never
writes it once — because the failure this rules out is the expensive one:
somebody drops CFG to 1 for a distilled model, raises it again, and their
negative prompt is gone.

A CFG that cannot be parsed leaves the row alone. Hiding a prompt box because a
number would not parse is the wrong way round.

Written as "at or below 1" rather than "exactly 1". Forge's own minimum is 1, so
the two are the same set today, and a build with a lower minimum then behaves
sensibly rather than surprisingly.


## 9. One box above the other, at every width

*Added by the Literal Prompt responsive layout spec (24 August 2026), whose own
sections 4–8 are the ones referred to here, and revised once the result was
seen on a real page.*

The spec asked for two boxes side by side while both stayed usable and one per
line when they did not, decided by the width of the row rather than of the
window — because Forge Neo's txt2img columns are dragged independently and a
maximised browser can hand this row 300px. That was built, and it worked:
`gr.Textbox(scale=1, min_width=240)` on each box, Gradio's Row being a wrapping
flex container, and measured in Chromium at side by side down to a 560px column
and stacked from 460px, live under the divider with no resize listener, no
window query and no theme detection anywhere.

It is not what the component does now. **The boxes always stack.**

Two reasons, both from watching it rather than from reading it. The first is
that a component whose height changes as the divider moves sits inside a prompt
column that keeps its own leftover space (§10.3), so the empty area below the
boxes grew and shrank while somebody dragged — the responsive behaviour was
visible mostly as the gap under it breathing. The second is `min_width` itself:
a component that states a minimum width is a component the column around it has
to be at least that wide for. That column belongs to the host and its width
belongs to the user.

So there is no `scale`, no `min_width`, no `flex-basis` and no width of any kind
in this component now. It is a `gr.Column` — Gradio's own word for "stack these"
— and the stylesheet says `flex-direction: column` and `flex-wrap: nowrap` on
both the container and the wrapper Gradio nests inside it, as a guardrail
against a theme with its own rules for Gradio's containers. Each box is
`width: 100%; min-width: 0`, and the wrappers between the block and the textarea
are `min-width: 0` too, because a flex child's default minimum is its content
and that is the usual way a prompt box ends up wider than the column holding it.

The id still says `row`. It is the name Python, `model_chain_literals.js` and
every test address this element by, and renaming it to match the layout would
be a rename of the contract to match a stylesheet.

### 9.1 The labels lost their explanations

`Positive Literal` and `Negative Literal`, with no `info=` copy and no
placeholder. The longer labels and the two paragraphs under them were written
for a component being met for the first time, and they read that way once:
after that they are two paragraphs of prose sitting between the native Negative
Prompt and the generation controls, in the one part of the UI that is other
people's. The explanation lives in the README, where it can be as long as it
likes.

The infotext keys and the settings keys keep the older wording —
`Model Chain Literal Positive` and friends — because those are a file format.
Relabelling a box is presentation; renaming a key is a migration.


## 10. What a browser said that reading the code could not

Rendered in Chromium against a repro of Forge Neo's own `ui_toprow.py` (Compact
prompt layout, Gradio 4.40, the host's `style.css` and this extension's loaded
together), with the host's real `extraNetworks.js` handlers on the page. Every
number below was measured there rather than reasoned about — which is the point,
because three of the four turned out to be the opposite of what reading the code
suggested.


### 10.1 The prompt family is not on `window`

`javascript/extraNetworks.js` says:

```js
let activePromptTextarea = {};
```

A top-level `let` in a classic script goes into the global **lexical**
environment. Every other classic script on the page shares it — so a bare
`activePromptTextarea` resolves from this extension's file — and it is *not* a
property of `window`. This file read `window.activePromptTextarea`, found
`undefined`, and quietly declined to register: no focus listeners, no membership
of the family, and every Extra Networks card going to the native positive prompt
however many literal boxes had been clicked into first.

Both forms are read now, bare first. The test that matters
(`test_a_host_that_declares_the_family_with_let_is_still_joined`) asserts the
`window` lookup is still `undefined` while the registration works, because the
two together are the whole of the bug.

The rest of §7 was right, which is what makes this worth recording: Forge Neo's
LoRA page sets `allow_negative_prompt = True`, and that is exactly the flag its
`cardClicked` reads to decide between the remembered box and the native prompt.
The feature was one identifier away from working.


### 10.2 Tag Autocomplete is offered, not assumed

Its `core` selector list does include `.prompt > label > textarea`, and in the
browser that selector matches both literal boxes — so the class was the right
mechanism and §8 was satisfied in principle. But that list is walked once, inside
its own `setup()`, and whether these two boxes are in the document by then is an
ordering neither extension controls.

So the two extensions are introduced from both directions, and neither
introduction needs to go first.

**Handed over.** `addAutocompleteToArea`, the same per-textarea entry point it
uses for its own late arrivals, which returns early on a textarea it has already
claimed. Feature-detected (`typeof addAutocompleteToArea === "function"`, and
`TAC_CFG` non-null, since it is `var TAC_CFG = null` until that extension's files
load) and retried on the next UI update.

**And asked for.** `getTextAreas` is a top-level function declaration in a
classic script, so it is a writable property of the global object and the call
inside that extension's `setup()` resolves through it. It is wrapped once,
additively: its list comes back with its own contents in its own order and these
two textareas appended. Whenever its setup runs — before this file, after it, or
in one of the re-scans it does for accordions that open later — the two boxes are
in the list it asks for.

That is what makes the ordering stop mattering, and why there is no timer here
waiting for that extension to finish loading. A page without Tag Autocomplete is
a page where both halves do nothing at all.

**And then neither half was the problem.** A user's log, once §10.5 existed to
produce one:

```
[config loaded, third-party boxes no, list extended True, boxes in the list True,
 row placed True]
```

Everything on this side done, and that extension refusing the boxes anyway. Its
**Active in third party textboxes** switch gates every textarea it does not
recognise as one of the four core prompt boxes — `getTextAreaIdentifier()`
compares against those four by identity and calls everything else
`.thirdParty.taN` — and these two are not four of them, however much they look
and behave like it. With that switch off there is nothing further to try: it is
the last thing between these boxes and tag completion.

It is also read in exactly one place. `grep activeIn` across that extension
finds `thirdParty` at one line: the gate at the top of `addAutocompleteToArea`.
Nothing re-checks it afterwards, so a textarea that gets past it once has
completion for the life of the page.

So the switch is lifted for the length of that one call, on these two
textareas, and put back exactly as it was — including `undefined`, on a build
where the option does not exist, because putting it back means putting it back.
Every other textbox that switch covers still answers to it, nothing else in that
extension's config is touched, and the line these boxes report says it happened:
a setting reading "off" that is not off for two boxes is not a thing to leave
somebody to discover.

Verified in a browser against that extension's real gate and real
classification, with the switch off: both boxes claimed, the switch still
`false` afterwards.


### 10.3 The empty space under the row was never ours

`ui_toprow.create_prompts()` builds `gr.Column(..., scale=6)`. In the classic top
row that 6 is a *width* share against the Generate column. In the Compact prompt
layout the same column is stacked inside the settings column, `ResizeHandleRow`
makes the settings and gallery columns items of a CSS **grid** — so the settings
column is stretched to the gallery's height — and the 6 becomes the largest claim
on that leftover height.

Measured with no extension on the page at all: a 989px prompt column holding
168px of prompt. 821px of empty space, in stock Forge Neo.

The Literal Prompt row lands in that column, so it inherits the gap — and while
the boxes still shared a line it changed how much of the gap showed as the
divider moved, which is how it came to look like something this feature had
done. It had not.

For one version this file wrote `flex-grow: 0` onto that column, which took it
from 989px to 194px and moved everything below it up. That worked, was guarded
three ways, and is **gone**. Reclaiming it meant reaching out of this component
and changing how somebody else's page lays out — for space that is the host's,
on an element the user may have their own theme's opinions about, and while the
user was asking for the opposite: leave the page's columns alone. The gap is
Forge's, the decision about it is the user's, and a line in their own `user.css`
settles it without an extension deciding for them:

```css
#txt2img_prompt_container.prompt-container-compact { flex-grow: 0 !important; }
```

`TestItLeavesTheHostsLayoutAlone` is the reason it does not come back: the only
element of the host's this file writes to is the Negative Prompt, one class, put
on and taken off again by §8's collapse.


### 10.4 And one thing this extension's CSS had wrong

Gradio groups adjacent form components into a `div.form` of its own, so the two
boxes are the row's **grandchildren**: `.mc-literal-row > .mc-literal-box` matched
nothing. The wrapping worked anyway — Gradio writes `min_width` as an inline
`min(240px, 100%)` on each box, and inline styles are what actually decided the
layout — but the defensive rule that was supposed to survive a theme overriding
that was inert. Both levels are addressed now, the wrapper by `*` rather than by
Gradio's name for it.


### 10.5 How the next one of these gets answered

Twice the answer to "tag completion does not work in these boxes" was a console
snippet somebody had to open the developer tools to paste. That is a fine thing
to ask of whoever wrote the snippet and a poor thing to ask of anybody else, and
it is the reason `mc_literal_report` exists.

The browser file builds a nine-field report the first time somebody puts the
caret in a literal box — which is exactly when tag completion is the thing being
expected — and posts it once to `/model-chain/literal-prompts/report`. That is
an endpoint inside the running WebUI and not a file: what it produces is one
line, in the console and in `<LLM data root>/logs/model_chain.log`, which
`mc_logfile` opens beside the `llama-server.log` the managed runtime writes.
(There was no such file before this: every line this extension logged existed
only in the terminal Forge was started from, which is no help to somebody
running it as a service or reading it the next morning.) A page where both boxes
were claimed says so; a page where they were not is a warning that names the
cause:

```
WARNING Model Chain: Tag Autocomplete is installed but has not claimed the
Literal Prompt boxes -- its "Active in third party textboxes" setting is off
(changing it needs a full restart, not a UI reload) [config loaded, third-party
boxes no, list extended True, boxes in the list True, row placed True]
```

Booleans and one word out of three, and no way for text to get in: `describe()`
reads a fixed set of keys, coerces every one of them, and ignores the rest of the
payload. The first version interpolated the payload's own `thirdPartyBoxes` value
into that line, which is how a string from the page reached the log —
`test_no_value_from_the_payload_reaches_the_line` caught it, and now every word
in the line is this module's own. A diagnostic that could carry what somebody
typed into a prompt box would be a diagnostic nobody should install.


## 11. Where the design intent was followed differently

**§10, reconstructing the fields on paste.** It does not happen automatically;
see §5 above. The intent's own word is *may*, and its own requirement that
reproduction not break is the one that decides it. The explicit restore button
does what the paste will not.

**§2, the row's position.** "Directly below the existing native Negative Prompt
row" is achieved by moving the row in the browser rather than by building it
there, because Forge offers no way to build it there. The failure mode is
benign and tested, but it is a DOM move and this repository is otherwise
careful not to make those — worth knowing about before someone reports the row
in the wrong place under an unusual theme.
