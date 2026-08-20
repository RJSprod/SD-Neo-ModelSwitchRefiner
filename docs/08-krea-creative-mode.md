# Krea Creative Mode — implementation notes

Companion to the *Krea 2 Creative Mode* design intent (20 August 2026) and its
data package, `krea2_creativity_library.zip`. That document states what the
feature is; this one records the choices made against it, the places it was
followed differently, and the handful of things that will otherwise be
rediscovered the hard way.

Section numbers below are the design intent's.


## 1. What happened to Krea Live

Krea Live was built against the previous design intent and is gone: the debounce,
the prompt-change observer, the generation-after-typing path, the continuous
reroll scheduler, the Live state machine, the Live timer settings and the
Generate-forever interaction are all deleted, along with their two test files
and their documentation (§11).

It is worth saying why the replacement is so much smaller, because the shape of
the difference is the lesson. Live's cache, revision counter, cooperative
cancellation of in-flight writes, latest-input-wins arbitration and failure
circuit breaker existed for one reason: work started without being asked for.
Every one of those mechanisms was answering a question that only exists if a
timer can fire. When every roll is explicit, all of them collapse into "the user
pressed it again", and roughly six hundred lines of Python and JavaScript go with
them.

One thing survived, and it was never really about Live: the arming token. Forge's
processing hook still needs to be handed a prompt computed before the image job
began, and it must still be impossible for a nested or queued generation to spend
that permission twice.

The other survivor is the ordering constraint (§4.1 below), which is a fact about
the broker rather than about either design.


## 2. What was built

| File | What it is |
| --- | --- |
| `prompt_master/krea/creativity/` | the vendored data package, byte-identical |
| `prompt_master/krea/CREATIVITY_LIBRARY_SOURCE.txt` | its provenance and digests |
| `prompt_master/krea/library.py` | loads and validates it; no UI, no inference |
| `prompt_master/krea/director.py` | the local Creative Director |
| `prompt_master/krea/variation.py` | reduced to the writer's sampling profile |
| `prompt_master/krea/enhancer.py` | accepts a finished brief for the user turn |
| `mc_llm_sessions.py` | one writer request, carrying the brief |
| `mc_creative_krea.py` | settings, roll history, the arming token |
| `scripts/model_chain_krea_creative.py` | the txt2img panel and the processing hook |
| `javascript/model_chain_creative_krea.js` | the explicit Generate gate |
| `mc_llm_krea_panel.py` | the same controls in LLM Studio |
| `mc_llm_progress.py` | the roll, reported on the host's progress bar (§7) |
| `tests/test_krea_creative.py` | the library, the Director, the scale, one-call |
| `tests/test_krea_creative_js.py` | the bypass and the absence of a scheduler |
| `tests/test_krea_progress.py` | the reporting, and the bar always being given back |


## 3. Where the design intent was followed exactly

- **The Director makes no model calls** (§1, §5). Asserted against its import
  graph rather than trusted: `director.py` imports nothing beginning `mc_`,
  nothing named for a client or an inference layer, and no network library. A
  planner call would have to add an import to get there.
- **One writer request per roll** (§9). Counted at every Creativity position, at
  every axis configuration, and across ten consecutive presses.
- **Creativity is semantic, not thermal** (§3, §10). Sampling climbs 0.60 → 0.96
  across the whole scale; the brief is what makes 10 different from 2. The
  previous design reached 1.24 and leaned on the sampler for difference it could
  not produce.
- **Creativity 1 is exactly legacy** (§3). Temperature 0.6, top_p 0.9, no extra
  sampler fields, *and* a user turn with nothing appended — the Director emits
  nothing below 2, so the message is byte-identical too. Checked as both.
- **Every Vary modifier scales** (§4). Four tiers per variant, and the
  one-axis-only case is tested directly: with Medium the only Vary axis, its
  expression still goes light → moderate → strong → extreme from 2 to 10.
- **Natural is absence** (§2). No line, not a hedged line. A brief saying
  "Texture: your choice" would put texture in the model's foreground, which is
  the opposite of leaving it alone.
- **Fixed survives Creativity 0 and 1** (§3). It is explicit configuration, and
  the scale governs variation.
- **The source prompt wins** (§7). Alias locks, plus the rule in words in every
  brief, plus the precedence order — source, Fixed, Vary, Natural — with a test
  that a lock beats a pin.
- **Seeds derive stably** (§6). SHA-256 rather than `hash()`, which Python salts
  per process; without that, "a fixed seed reproduces" would be true only within
  a session. The exact derived values are pinned in a test.
- **Krea's instruction is untouched** (§9, §14). Every word of art direction
  travels in the user turn. A test asserts the system message is identical at
  Creativity 1, 5 and 10.
- **The processing hook applies and never requests** (§11). It has one verb:
  `consume`.


## 4. Two things worth knowing before changing this

### 4.1 The model must be asked before the image job starts

`mc_llm_sessions` takes the broker's workload lock for a whole run and waits
while `mc_broker.host_busy()` is true. A roll requested from inside a Forge
processing hook would therefore be an LLM run waiting for the image job that is
waiting for it. This is why the shape is *roll in a Gradio handler, apply in the
hook*, and why the browser has a bypass flag at all: the roll has to finish
before the native click is allowed through.

### 4.2 The arming token is consumed, not checked

`consume()` takes the token and clears it in one locked step, so one roll
produces exactly one image. That matters for more than tidiness: Stage 2's own
nested `process_images()` call, a queued request from a closed tab, and a second
Generate click during a batch would all otherwise inherit permission granted
once.


## 5. Where it was followed differently, and why

### 5.1 The user turn keeps this package's own labels

§9 suggests `USER REQUEST:` and `CREATIVE DIRECTION:`. The implementation uses
`user_prompt:` and `creative_direction:`, because `enhancer.py` already labels
every block in the user turn that way — `user_prompt:` and `reference_images:`
were there before Creative Mode and are covered by existing tests. A third block
shouting in capitals would read as though it came from somewhere else, and the
labels are this extension's to choose. The *content* of the block is the design's,
including the source-priority sentence verbatim in spirit.

The design's trailing "Expand this into one Krea 2 prompt." is also omitted: the
vendored `expansion.txt` already instructs exactly that, and repeating it in
every user turn would be this repository restating upstream's instruction in a
place upstream cannot see.

### 5.2 The expanded prompt is not recorded in infotext

§15 lists what to record per image. Everything on that list is recorded except
the expanded prompt, which is not missing — it *is* the image's own `Prompt:`
line, because that is what Creative Mode substituted before sampling began. A
second copy under a `Krea Expanded Prompt` key would add a few hundred bytes to
every PNG in order to repeat what the file already says, and would create a copy
that a later paste could disagree with. `Krea Pinned LoRAs` is recorded instead,
which is the only difference between the writer's paragraph and the prompt that
was generated from.

### 5.3 One settings file for both surfaces

§15 lists the preferences to persist without saying whether the two surfaces
share them. They share all of them, including the Creative Mode toggle. The
axes, the Creativity position and the seed describe *how this installation does
art direction*, and somebody who has spent five minutes configuring ten axes in
LLM Studio should not have to do it again in txt2img.

Sharing the toggle is the arguable half. It means enabling Creative Mode in LLM
Studio also changes what txt2img's Generate does — but the txt2img checkbox shows
that state plainly, and the alternative is two toggles that look identical and
mean different things. If this turns out to be the wrong call in use, splitting
it is one preference key and two lines.

### 5.4 Axis activation is weighted, not uniform

§4 gives the active-axis *count* per position but not which axes. A uniform draw
would make Creativity 2 activate "detail emphasis" alone about as often as
"medium" alone, and one axis of direction should be the axis that changes the
picture. `library.priority_of()` supplies a default weighting — medium highest,
then style and lighting — under which medium is roughly twice as likely as lens
at Creativity 2.

That weighting is *not* in the data package, which is where it belongs long term,
so the loader reads an `axis_priority` map from the manifest when a later package
carries one and falls back to the built-in default otherwise. No code change is
needed to override it.

### 5.5 Compatibility preferences are a boost; avoidances are absolute

§8 says the Director should "reject or resample obviously incoherent
combinations". Both halves of `compatibility.json` are honoured, but differently:
`avoid_*` rejects a candidate outright, while `prefer_*` doubles its weight.

A hard preference would empty the candidate pool whenever the preferred partner
had not been drawn yet — the axes are visited in one order, so requiring a
painting medium *before* the medium is chosen would silently drop the texture
axis. Rejection is checked in both directions (a rule on the candidate against
what is chosen, and a rule on what is chosen against the candidate) so coherence
does not depend on visit order.

### 5.6 Anti-repetition falls back rather than failing

At Creativity 10 the penalty is total, so a candidate pool made entirely of
recent choices sums to zero weight. When that happens the unpenalised weights are
used instead. "Strongly avoid repeats" must never become "refuse to direct this
axis at all once the library runs short", and a user who has rolled thirty times
should still get a medium on the thirty-first.

### 5.7 Fixed axes use the light tier at Creativity 0 and 1

§3 says Fixed still applies at 0 and 1; §4 says Fixed uses the tier corresponding
to Creativity. At 0 and 1 the policy's tier is `"none"`, which names no
expression, so the two rules do not quite meet. Fixed renders at `light` there —
the least-amplified wording available, which matches what those positions mean
without inventing an intensity the package does not define.

### 5.8 The vendored client edit was reverted

The previous design's §8.2 asked for `prompt_master/inference/llama_client.py` to
grow an optional whitelisted sampling parameter, and it did. This design's
sampling table has no optional fields at all — semantic direction does the work —
so that edit is dead code, and it has been reverted. `prompt_master/` is
byte-identical to its upstream again, and `VENDORED_FROM.txt` is back to saying
so truthfully.


## 6. The data package

`prompt_master/krea/creativity/` is copied whole from the delivered ZIP with no
edits, digests recorded per file. `examples/` and `tests/` are not loaded at run
time and are kept anyway: they are what a later version will be diffed against,
and a vendored package that dropped half its files is one nobody can update
cleanly.

Two rules from the package's own README are enforced in code rather than trusted:

- **Stable ids.** A saved Fixed selection stores a variant id, so renaming one
  silently changes what a user's configuration means. `library.py` refuses a
  package with a duplicate id within an axis; renames across versions are a
  contract this repository cannot enforce and the provenance file says so.
- **All four tiers.** A missing tier is refused at load rather than substituted
  from a neighbour, because a silently substituted tier is a Creativity slider
  that stops scaling on one axis and says nothing about it.

The sampling table is deliberately duplicated between `creativity_policy.json`
and `variation.py`, with a test asserting they agree. Creativity 0 and 1 are
compatibility guarantees, and a data package that failed to load — or was edited
by somebody tuning the creative vocabulary — must not be able to move them.


## 7. What a roll costs, and how it says so (21 August 2026)

The first thing anybody noticed after Creative Mode landed was that Generate
took twenty seconds to do anything, with no sign that it was working. Both
halves of that turned out to be worth writing down.

### 7.1 Where the twenty seconds go

From a llama.cpp server log of a real session, one Creativity-10 roll on a 26B
mixture-of-experts model in Mixed placement:

```
params_from_       request arrives
   +0.16s          prompt-cache housekeeping
restored context checkpoint (pos_max = 460)      <- Krea's instruction, cached
erased  invalidated checkpoint (pos_max = 841)   <- last roll's brief, useless
   +10.3s          prompt eval, 355 tokens at 29 ms each
   +12.4s          generation, 179 tokens at 69 ms each
             total 22.7s
```

Two of those lines are the whole explanation. The server can reuse its cached
prefix exactly as far as the end of the system prompt, because **the creative
brief is different on every roll** — so the previous roll's checkpoint is
*erased as invalidated* and several hundred tokens are evaluated fresh, every
time. Before Creative Mode the user turn was ~19 tokens and prompt evaluation
took 0.66 s.

The brief's length is the lever: measured over the shipped vocabulary it grows
from ~240 characters at Creativity 2 to ~2,900 at Creativity 10, because the
extreme tier expresses all ten axes in full. Placement is the multiplier: at
~36 tokens/sec of prompt evaluation, every token of brief costs about 27 ms, and
a resident model costs a small fraction of that.

So the cost is real, proportional to Creativity, and not a defect. What was a
defect was that none of it was visible.

### 7.2 One thing worth fixing in the data package

Every ``extreme`` expression ends with the same clause — *"push this treatment
far enough to define the visual language while preserving the user's subject and
explicit constraints"* — and at Creativity 10 that clause is repeated once per
axis, nine or ten times, for roughly 200 tokens of literal duplication. The
brief already states the preserve-constraints rule once, at the top, in
``director.SOURCE_PRIORITY_RULE``.

It is not fixed here, and deliberately: those strings are in the vendored data
package, whose digests are recorded and whose whole point is that it can be
re-vendored cleanly. Rewriting them at assembly time would be this repository
editing vendored text at run time, which is the thing ``prompt_master/krea``
exists to make impossible. It belongs in a 1.0.1 of the library — where it is
worth about five seconds a roll at Creativity 10.

### 7.3 The roll reports itself on the host's bar

``mc_llm_progress.py``. The design is to use the host's machinery rather than
draw anything:

- the server mints a task id and sends it down the hidden box the browser is
  already polling for the arming token, as a ``task:`` message;
- the browser calls the host's ``requestProgress`` for that id, which is what
  draws and polls the real bar;
- the server claims that id with ``modules.progress.add_task_to_queue`` and
  ``start_task``, which is what ``modules.call_queue`` does, and releases it with
  ``finish_task``;
- ``shared.state.textinfo`` carries the phase name and ``sampling_step`` /
  ``sampling_steps`` carry streamed characters, so **the existing** ``mc_progress``
  arithmetic describes the roll with no special case anywhere.

The three phases are the ones llama.cpp's log distinguishes and the ones a user
feels differently: *Waiting* (GPU handover, or a cold llama-server), *Reading*
(prompt evaluation — reports nothing, and is the phase Creative Mode made long),
*Writing* (streams, so it is the only one that can say how far through it is).

Characters stand in for tokens throughout. Nothing on this side of the wire has
a tokenizer, and the error is a constant factor that the calibration store folds
in on the first measurement and never sees again.

Three details that are not obvious:

- **The reply length is learned, with a floor.** It sizes the writing phase, and
  one unusually terse answer would otherwise teach the bar to expect eighty
  characters and leave every normal roll pinned at 99%.
- **The denominator stretches.** A reply that outruns the estimate grows
  ``sampling_steps`` rather than pinning the bar at full — ``mc_progress``
  already refuses to go backwards, so the bar slows instead of rewinding.
- **The task is released on every path out.** A claimed task that is never
  released leaves the host believing a job is running, and the next Generate
  draws a bar that never moves. ``tests/test_krea_progress.py`` checks the
  release after success, a model that throws, an empty reply, a refused
  checkpoint, an empty source, a Director failure and an Interrupt.

Interrupt is wired because the bar carries the button whether or not anything is
listening, and a button that does nothing is worse than no button. The flag is
cleared on the way out so the image generation that follows does not inherit a
stop nobody meant for it.

### 7.4 Two things the first attempt at that got wrong

Both were reported from a real machine on the day it landed, and both are worth
keeping written down because each is the obvious thing to do.

**The bar deadlocked the run it was drawing.** Claiming the task also set
``shared.state.job`` and ``shared.state.job_count``, the way a real job does.
But ``mc_broker.host_busy()`` *is* "either of those is truthy", and
``mc_llm_sessions._Gpu.acquire`` refuses to start while it is true — so the
language model sat waiting for an image generation that was itself, printing
*"Waiting for image generation…"* on a machine generating nothing, while the bar
crept to its 0.999 ceiling and stopped. This is precisely the deadlock §4.1
describes, reintroduced through the back door by a progress indicator.

It was intermittent, which made it worse: any Gradio call finishing nearby
clears both fields in its own ``finally``, so it hung on a fresh restart and
sometimes escaped a few seconds later having wasted the wait. Neither field is
set now, and ``TestTheBarNeverBlocksTheRunItDescribes`` drives the whole txt2img
gate with the real ``host_busy`` — not the stub every other test in that file
uses — and asserts it answered False at every frame.

The cost is that the host's own progress arithmetic needs ``job_count`` to
compute a fraction, so with this extension's whole-job reporting switched off
the bar shows the phase name without moving. That is a smaller loss than a hang.

**The task id came in through a Gradio ``js=`` hook.** That mirrored the host's
own ``submit()``, and it made the roll depend on a Gradio contract — what a
``js=`` function's return value means — that this extension had never relied on
before and could not verify. The id now travels the other way: the server mints
it and sends it as the first frame down the channel the browser was already
polling. Nothing has to arrive intact for the roll to work, so a bar that cannot
be drawn costs a bar rather than a generation, and the ``js=`` question does not
arise.

### 7.5 Creative Mode inverted the load order, and the card noticed

The third field report, and the one with the clearest evidence in the llama.cpp
log. Free VRAM at each `llama-server` start, across one user's log:

```
sessions 35-38:   3078, 2963, 2868, 2766 MiB free    <- image model already loaded
sessions 39-52:  23304 MiB free, every time          <- empty card
```

That is the whole bug. Before Creative Mode the order was *image first, language
model second*: the checkpoint was loaded by generating, and `negotiate` sized the
LLM into the three gigabytes that were left — which is exactly what it is for,
and why both fitted. A Creative roll reverses it. The language model now loads
first, onto a card with nothing on it, llama.cpp's own `-fit` sizes it to
everything it finds, and the checkpoint that has to run three hundred
milliseconds later gets the remainder. On a 24 GB card that is the difference
between "both fit" and "the image model does not".

Nothing leaked. The extension's own reclaim path did not save it either, because
that path is only reached when *this extension* loads a checkpoint: for an
ordinary single-model generation Forge loads it directly, so `mc_memory`'s
eviction and the broker's `_reclaim_for_image` hook are never in the call chain.

The fix is the one `mc_llm_runtime.Runtime.release` already points at in its own
docstring — *"which is why `negotiate` works hard not to have to ask for image
VRAM in the first place"*. `negotiate` has always taken an `extra_reserve`; it
had only ever been used as an out-of-memory retry penalty. A Creative roll now
passes the image pass's requirement through it:

```
mc_creative_krea.image_reserve_bytes()      # what the pass needs, minus what
    -> sessions.krea(..., reserve=)          # the image family already holds
    -> sessions._client(needs_vision, reserve)
    -> runtime.client(needs_vision, reserve=)
    -> negotiate(..., extra_reserve=reserve)
```

Three details worth keeping:

- **What the image side already holds is subtracted.** Those bytes are the
  loaded checkpoint. Reserving them a second time would shrink the language
  model to make room for a model that is already there.
- **Only the txt2img path reserves anything.** `guard_checkpoint` is that path
  and only that path, which makes it exactly the condition under which a picture
  follows. LLM Studio writes a prompt and stops; reserving image VRAM there
  would shrink the writer for a picture nobody asked for.
- **Leaving room is much cheaper than reclaiming it.** A running llama-server
  can only give VRAM back by stopping, so a reserve that is right costs nothing
  and a reserve that is missing costs a restart per generation.

`hand_back_vram()` is the second half, and it is recovery rather than
prevention: a server that was already up and holding the card when the reserve
was introduced cannot shrink in place, so the roll asks the broker for the room
before handing the prompt over. It is a no-op in the ordinary case —
`request_vram` returns immediately when what is free already covers the
requirement — so a correctly sized card never pays for the call.

### 7.6 Why not just cache the brief

Because the brief *is* the variation. Reusing one would mean successive presses
produced the same art direction, which is the feature working backwards. The
cheap wins are the ones the README lists — resident placement, a lower
Creativity, more Natural axes — and each of them shortens the brief or the
per-token cost rather than defeating the point of it.


## 8. Tests

`tests/test_krea_creative.py` is mostly measurement. The Director's promises are
properties of a distribution rather than of a function, so they are checked over
hundreds of rolls: a dozen distinct mediums reachable from a bare "car" at
Creativity 10, a stated medium never replaced across forty rolls, impasto texture
never landing on a photographic medium across a hundred and twenty, recent
variants avoided outright at 10 across two hundred.

The rest is the counter. `FakeClient.calls` gets one entry per completion
llama.cpp was asked for, and every assertion about it is `== 1`.

The package's own `acceptance_cases.json` is read rather than paraphrased, and
`TestTheAcceptanceCases` fails if the package grows a case nothing claims — which
is what will stop the data and the tests drifting apart the next time the library
is updated.

`tests/test_krea_creative_js.py` runs the browser gate under node against a
synthetic clock, which is the only way to ask the two questions that matter
there: does the one-shot bypass let exactly one native click through, and is
there really no scheduler left. The second is asked by running an hour forward
with nobody touching anything and asserting that no timer remains armed.

`tests/test_krea_progress.py` covers the reporting, and is mostly about the bar
being *given back*. It also drives the submit hook under node against a fake
page, because "the id reaches argument zero and the host is asked to draw a bar
for that same id" is two facts about one function and neither is visible from
reading it.
