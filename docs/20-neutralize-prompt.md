# Neutralize Prompt — implementation notes

Companion to the *Neutralize Prompt* design intent and technical specification
(4 September 2026) and its revision notes. That document states what the stage
is; this one records the choices made against it, the places it was followed
differently, and the handful of things that will otherwise be rediscovered the
hard way.

Section numbers below are the design intent's.


## 1. What this is, in one sentence

A language-model pass that may delete, checked by a function that proves it did
nothing else.

That sentence is the whole design, and the second half of it is the half that
matters. The instruction in `prompt_master/krea/neutralize.txt` asks a model to
remove pose geometry and image placement and to add nothing. An instruction
cannot check; `neutralizer.subtraction_error()` can, and the pipeline trusts
the reply because the guard found nothing in it the source did not have, in
the order the source had it — never because the instruction was clear. A reply
that fails is refused whole and the generation carries on from the prompt as
typed. Refusing whole rather than repairing is deliberate: deleting the one
added word would be a second editor nobody can see, running after the one
whose instruction is on disk.

Everything else is the existing machinery, entered from one more place: a
third role in `mc_llm_roles`, a third session generator in `mc_llm_sessions`,
a third `Pass` in `mc_llm_progress`, a fourth row in `mc_pipeline_panel`, and
one more phase in `mc_krea_pipeline.run()`.


## 2. §2.3 — `working`, and why the Composer reads it

`mc_krea_pipeline.run()` used to have three texts: source, scene, final. It
now has four, and the new one sits between the first two:

```
source  = the transformable text, literals already out
working = source, or what the Neutralizer left of it
scene   = working, or what the Creative Writer made of it
final   = scene, or what the compositor made of it
```

Every stage after the Neutralizer reads `working` where it used to read
`source`, and the Smart Composer is the case that decides the question. It is
handed a *source* to reconcile the scene against, and handing it the
pose-heavy original would reintroduce, as a comparison, exactly the constraints
the Neutralizer took out — the Composer would dutifully put "centred in frame"
back because the source it was shown still said so. So `compose(source=working)`,
and `tests/test_krea_neutralizer.py` asserts what the Composer's user turn
contains under Neutralize + Smart.

The metadata is the one reader that keeps the original: `Krea Neutralize
Source` records `request.raw_source`, brackets and all, because a restore that
put the neutralized text back in the prompt box would be restoring the
pipeline's answer rather than the user's question.


## 3. §3.6 — the guard, and what "a word" is

`_WORD = [^\W_]+`: a run of letters and digits, casefolded. Everything else —
spaces, commas, hyphens, apostrophes, underscores — is a separator and
contributes nothing. That is what lets a reply repair the punctuation around a
deletion without being refused for it, and it is also exactly as wide as the
normalisation goes: `silver-haired` and `silver haired` are the same two
tokens, `woman's` and `womans` are not, and a spelling change is an edit this
pass was not asked to make.

The check is an in-order subsequence walk with `list.index(word, position)`.
Two failures are told apart — a word the source never had, and a word the
source had but out of place — because the console can say which kind of run
it was. **The word itself is never in the reason.** A moved word is a word of
the prompt and an added one is the model's guess at the prompt, and the rule
for every line this extension logs is that it says what kind of run it was and
never what was in it. The first draft quoted the offending word; the test
`test_a_refusal_never_quotes_the_word` is why it no longer does.

The guard is one-sided, and the module docstring says so where it will be
read: it proves nothing was added or moved and cannot prove too much was
taken. Over-deletion is the instruction's job, and the acceptance fixtures in
`tests/test_krea_neutralizer.py::TestTheAcceptanceCases` check it against the
same wording the model is given, so the two cannot drift apart.


## 4. §3.7 and §12.2 — failed, stopped, and the flag that is left alone

`mc_neutralize.Neutralized` has two exits that are not success, and they are
different types of thing. `failed` is a reason the source stays as typed —
no runtime, no model, a refused reply, an exception, nothing back — and the
pipeline's answer to every one of them is the picture from the prompt as
typed, with a comment on the image saying what did not run. `stopped` is the
user's Interrupt, and the pipeline's answer to that is `Outcome.cancelled`:
`run()` returns before the writer, the compositor and the hand-back, and the
hook returns before it says what ran.

The driver reads the Interrupt off the host's own bar, once per event, exactly
as the Composer does. What it does *not* do is clear the flag: the bar is the
generation's, and the host reads that same flag a moment later to stop the
sampling. A driver that tidied up after itself would turn the user's Stop into
a generation that went ahead — which is the one conversion §12.2 forbids.

There is one more stop path, and it is worth knowing it is defensive. The
session yields `CANCELLED` when its `Cancellation` is set, and the driver owns
that object, so in practice every route the driver has to a `CANCELLED` goes
through the host flag first. The branch that maps `CANCELLED` to `stopped` is
still there because it is the session's declared contract, and it is tested
by driving the driver with a session that yields the event directly — the
mutation check found that the flag-based test alone never reached it.


## 5. §4 — three roles, and what "the other role" became

`NEUTRALIZER` is first in `ROLES`, because that is the order the phases run
in and every "for role in ROLES" now enumerates in execution order. The
inheritance, the split, the setup selector and the Performance settings are
the existing machinery keyed by one more name; nothing role-specific was
written for it.

What did need writing was every sentence that assumed two. `_role_line` in
`mc_llm_studio` used to name Creative and Spatial by hand and speak of "the
other role", which stopped having a referent the day a third one arrived. It
now asks the registry: `partners(role)` for who shares a runtime with this
role, `contending()` for the first memory pool holding more than one distinct
runtime, and `mc_llm_roles.describe()` for "A, B and C" in prose. The
`SHARING_MODES` labels and the two settings titles in `scripts/model_chain.py`
were reworded the same way — "before starting another's" rather than "the
other's". `tests/test_krea_neutralizer.py::TestTheThirdRole::test_nothing_reasons_about_a_partner_any_more`
pins `others()` and `describe()` to their three-role answers, so a helper that
went back to answering for a pair fails there before it fails on a screen.

The five partitions of §14.3 are five tests, and the setup that made them
honest is worth recording: in the first draft the Neutralizer inherited
Creative's configuration object, so a test that patched "the Creative role's
config" was patching the Neutralizer's too and the all-shared case passed for
the wrong reason. `tests/test_llm_roles.py` grew a `trio()` helper beside
`pair()` that builds three distinct configurations, and the partition tests
use it.


## 6. §6 — the invariant, and where it had to live

§6.2 says the Neutralizer must opt out of any authority to reclaim the image
side "while still sharing the rest of the role configuration". On the branch
that authority already existed: **Memory priority → LLM priority** is exactly
permission for `_make_room_for_the_llm()` to release image residency, and a
Neutralizer configured to follow the installation inherits whatever the
installation set.

The obvious fix — override `memory_priority` in the Neutralizer's resolved
config — is wrong in a way that would not show up until a user had two
servers where they expected one. `RuntimeRegistry` coalesces roles by
`_identity()`, and `memory_priority` is part of the identity: a role that
differs only there is a different server. Forcing the Neutralizer's priority
to Cooperative would split it from a writer set to LLM priority, and the
sharing the spec spends a section on would silently stop for the one setting
that exists to make the writer faster.

So the opt-out is per request, not per role. `Runtime.client()` takes
`image_reclaim=True`, threads it to `_make_room_for_the_llm(allowed=...)`,
and `sessions._client()` passes it through; `_neutralize` is the only caller
that passes `False`. The server is placed exactly as the writer's would be —
same identity, same ladder, same reserve — minus the one eviction a writer set
to LLM priority is allowed. `sessions._client` only forwards the keyword when
it is `False`, so every existing test double of `runtime.client` keeps
working.

The test proves behaviour rather than configuration, as §6.2 asks:
`TestTheImageModelIsNeverEvicted` builds a registry under LLM priority with a
GGUF header too large for one gigabyte of free VRAM, spies on
`mc_broker.release_for_llm` and `request_vram`, and asserts the writer's
request reaches the reclaim and the Neutralizer's, same registry, same
setting, does not — and is placed anyway, down the ladder. A companion test
reads `mc_neutralize.py` and the session for any name that could reach an
image reclaim, so the invariant cannot be re-entered by a helper.


## 7. §7.3 and §7.4 — the hand-back is the request's to decide

The Writer used to ask "does a Smart Composer follow?" before handing the card
back. With three phases that question has three askers and six answers, and
§7.4 asks for one owner instead. `Request.llm_phases` is a tuple built once
from the switches — `(neutralize, write, compose)` filtered to the ones asked
for — and `last_llm_phase` is its last element. Each phase in `run()` hands
back only when it is that one. Adding a fourth phase means adding it to the
tuple, not to three `if` statements.

`WRITE` is listed when Creative Mode is on whether or not the hook supplied a
writer, because the phase is what the request asks for; a writer that then
does not run is a failure the existing ladder already covers, and a failed
last phase still hands back — the server it left behind is the same server.

`TestTheHandoff` records how many client calls had been made when
`hand_back_vram` was called, which is how "after the Neutralizer" and "after
the Composer" are told apart without timing anything.


## 8. §9 — a row with nothing to open

The three existing rows are Accordions because each has a body. This one has
none, and §9.4 forbids an empty drawer with a dead caret, so
`mc_pipeline_panel` grew a second row kind: `PLAIN` beside `EXPANDABLE`, with
`OWNED` still the full ordered tuple every consumer iterates.

A plain row is a stage Column, already carrying the `CARDED` class from the
server rather than waiting for the browser file to add it, a `gr.HTML` holding
the two lines, and the same switch Column in the same lane. The two lines are
written by `plain_label()` as the *same markup* the browser file produces when
it splits a card's label — `.mc-pipeline-card-head > .mc-pipeline-label >
.mc-pipeline-name + .mc-pipeline-said` — so the stylesheet's existing rules
for a name and a description apply unchanged and there is no second set to
keep in step. `card_summary()` returns a `value` update for a plain row and a
`label` update for a card, and the script never has to know which it is
talking to.

One rule from §10.1 of the pipeline notes carried over unchanged: the switch
is the only control and the only source of truth. There is no hidden checkbox
mirrored by a visible one.

The stylesheet block is small and every line of it earns its place: the HTML
wrapper is stripped of Gradio's padding, border and background; the head is a
block of the shared `--mc-pipe-head` height with the same lane-plus-caret
padding on its right — the caret zone is reserved even though nothing is drawn
in it, so the four rows' text columns line up — and the label is a block at
full width. `tests/test_pipeline_render.py` renders it in the real browser
beside the three cards and measures the same things: two stacked lines, the
name larger and heavier, nothing under the switch, nothing but the label in
the head, and no caret content.


## 9. §9.8 — the argument tuple, and a bool that has to be a bool

`ScriptKreaCreative.ui()` returns
`[enabled, creativity] + settings + axes + literal(2) + neutralize(1) + spatial(3)`.
The Spatial tail stays at the absolute end because `mc_plan.creative_from()`
reads it from there; the new field goes immediately before it, and
`NEUTRALIZE_CONTROLS = 1` names its width. `_split()` recognises the full
shape, the shape before this field, the shape before the literal fields, and
the no-panel shapes, and returns `neutralize` as a fifth element that is
`False` whenever the field is absent.

`mc_plan.neutralize_from(p)` reads the same position off the tail, and it
returns `isinstance(found, bool) and found` rather than `bool(found)`. That is
not fussiness. The position it reads is *inside* an older caller's tuple —
under the previous shape it is the last axis value, a string — and a truthy
string there would arm a language-model stage on behalf of an API caller that
never heard of it. §9.8's "absent field means False" is only true if a
present-but-foreign value also means False.


## 10. §11 — the two keys, and when they are written

`Krea Neutralize Prompt: True` and `Krea Neutralize Source` are written only
when the pass ran and its answer was used. A switched-on toggle whose pass
failed records nothing, because the picture was made from the prompt as typed
and a record saying otherwise would tell a later paste to switch off a stage
that never touched it. An unchanged reply *is* a run and is recorded as one —
the revision notes' seventh point — because success means the stage completed
and its accepted answer was used, not that text necessarily changed.

The paste field is the Creative one's twin: the flag's presence switches the
stage off and its absence leaves the switch alone. `_restore_setup` returns
seven outputs now — the switch is last — and a neutralize-only image restores
its typed source into the prompt box from `Krea Neutralize Source`, since
there is no `Krea Source Prompt` when Creative Mode did not run.


## 11. §10 — one more Pass, and its baselines

`mc_llm_progress.NEUTRALIZER` is a `Pass` with three phase keys of its own
(`krea:neutralize:read|write|reply`) and the three labels the browser file
matches on. The labels are deliberately not the writer's: "Waiting for the
language model" is a phase two rows could claim, and §9.9 forbids the browser
from guessing which. `mc_progress.BASELINES` seeds the read and write rates
from the writer's — same server, same kind of work per token — and gives the
reply its own, smaller allowance, because a reply that is a subset of a prompt
box is a fraction of an expansion. The measured values replace all three after
the first run, as every other phase's do.

The pipeline's JavaScript grew the three matches and a four-element `ORDER`,
and `tests/test_krea_neutralizer.py::TestTheRowFollowsTheBar` drives it under
node: every Neutralizer label lights the Neutralize row, no writer label does,
and a bypassed row is never marked done by a generation that never entered
it.


## 12. Where the design intent was followed differently

* **§6.2 — the opt-out is a request keyword, not a role setting.** Explained in
  §6 above: a per-role override would change the runtime identity and split a
  server the user meant to share.
* **§7.2 — three phase labels, not four.** The spec lists "Preparing the
  prompt neutralizer..." as well. The session already emits the shared
  preparing line through `_preparing(role)`, which carries the role's own
  prefix, and the driver counts it as part of the wait — the pass's own
  announcement, emitted immediately before the request goes out, is what
  starts the reading phase. A fourth label would have been a fourth thing for
  the browser file to match with nothing new to light.
* **§9.4 — `gr.HTML` rather than a text component.** The spec suggests "a small
  text/summary component". Markdown and Textbox both wrap their content in
  chrome a theme owns; an HTML block holding the card's own markup is the one
  choice that lets the existing name-and-description rules apply verbatim.
* **§13.3 — no new failure notice.** An unconfigured installation fails the
  pass the same way an unavailable server does, and the comment on the image
  and the console line point at LLM Studio → Setup as every other role's
  failure does. Nothing Neutralizer-specific was added.


## 13. Two things the tests taught

**A same-length mutation in the same second leaves a stale `.pyc`.** The
mutation harness that checks each new invariant writes a mutated source,
runs pytest, and restores the original. Python's bytecode cache invalidates on
source mtime and size; a mutation that changes neither — `Neutralize` to
`Neutralise` — restored within the same second leaves the *mutated* bytecode
in `__pycache__`, and the next honest run fails on code that is no longer on
disk. Clear `__pycache__` after a mutation pass, or make the mutation change
the length.

**A test that asserts `stopped` must own the flag.** `FakeState.interrupted`
is reset by the `host` fixture and nowhere else. A driver test that set it to
prove a stop, followed by one that did not take `host`, saw the previous
test's flag and passed for the wrong reason — the mutation check on the
`CANCELLED` branch was what noticed. Tests about the Interrupt take `host`,
and the one that drives the session-level stop asserts the flag is clear
before it starts.
