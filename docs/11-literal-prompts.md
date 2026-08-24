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

**The one thing not verified here.** §8 asks for Tag Autocomplete and LoRA
autocomplete in the two new fields. They carry the host's own `prompt` class,
which is the pattern those extensions match on, and they are stock Textboxes
with nothing custom about their text entry — but no test in this repository
runs Tag Autocomplete, so this is a well-founded expectation rather than a
checked one.


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


## 9. The row is a flex wrap, not a media query

*Added by the Literal Prompt responsive layout spec (24 August 2026), whose own
sections 4–8 are the ones referred to here.*

Forge Neo's txt2img columns are dragged independently, so the width of the
window is not the question being asked: a maximised browser can hand this row
300px, and a media query keeping two boxes side by side in it pushes them into
the image column. The width that matters is the row's own.

That makes it a flex problem rather than a JavaScript one. A Gradio Row is a
wrapping flex container, and `min_width` on its children is the width below
which they stop sharing a line — so the entire responsive behaviour is two
`gr.Textbox(scale=1, min_width=LITERAL_BOX_MIN_WIDTH)` calls. It follows the
divider live because layout does, with no `ResizeObserver`, no window listener,
no theme detection and no knowledge of Forge's column ratio anywhere.

`LITERAL_BOX_MIN_WIDTH` is 240 — roughly a prompt box's worth of usable text —
and `style.css` states the same number again in `.mc-literal-box`, because a
theme is free to restyle Row and the guardrail is worth more than the
duplication. `TestTheRowFitsItsColumn` asserts the two are equal, so tuning one
fails until the other is tuned with it.

The stylesheet's half is `width`/`max-width: 100%` with `flex-wrap: wrap` on the
row, `flex: 1 1 240px` with `min-width: min(240px, 100%)` on each box, and
`min-width: 0` on the wrappers Gradio nests inside them — a flex child's default
minimum is its content, which is the usual way a prompt box ends up wider than
the column holding it. `min(240px, 100%)` rather than `240px` so that a column
narrower than one box takes the box down with it instead of being overflowed.
Nothing here is given a pixel width, and nothing here can start a horizontal
scroll bar.

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


## 10. Where the design intent was followed differently

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
