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

One thing survived Live and did not survive what came after it: the arming
token. It was permission for exactly one native generation to spend a prompt
computed before that generation began, and §9 is the change that removed the gap
it guarded.

The other survivor is the ordering constraint (§4.1 below), which is a fact about
the broker rather than about any of these designs — and §9 is where it is finally
answered rather than routed around.


## 2. What was built

| File | What it is |
| --- | --- |
| `prompt_master/krea/creativity/` | the vendored data package; every file byte-identical but `defaults.json` (§10.2) |
| `prompt_master/krea/CREATIVITY_LIBRARY_SOURCE.txt` | its provenance, digests and the one edit |
| `prompt_master/krea/library.py` | loads and validates it; no UI, no inference |
| `prompt_master/krea/director.py` | the local Creative Director |
| `prompt_master/krea/variation.py` | reduced to the writer's sampling profile |
| `prompt_master/krea/enhancer.py` | accepts a finished brief for the user turn |
| `mc_llm_sessions.py` | one writer request, carrying the brief |
| `mc_creative_krea.py` | settings, roll history, one roll, and what a paste left behind (§10) |
| `mc_creative_panel.py` | the control surface, built once for both surfaces (§10.3) |
| `mc_creative_profiles.py` | named configurations and the chosen default (§10.4) |
| `mc_infotext.py` | the Creative paste fields and the configuration keys (§10.5) |
| `scripts/model_chain_krea_creative.py` | the txt2img panel, and the hook that rolls and applies |
| `javascript/model_chain_creative_krea.js` | the armed indicator, and nothing else (§9) |
| `mc_llm_krea_panel.py` | the same controls in LLM Studio |
| `mc_llm_progress.py` | the roll, reported on the host's progress bar (§7) |
| `tests/test_krea_creative.py` | the library, the Director, the scale, one-call |
| `tests/test_krea_creative_js.py` | that the click is never held, and the absence of a scheduler |
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
- **The processing hook is where the prompt is settled** (§11). It used to only
  apply — one verb, `consume` — because requesting from inside a running job
  deadlocked. §9 removed the deadlock, so it now requests and applies, and
  nothing else in the feature can.


## 4. Two things worth knowing before changing this

*Both of these describe the design as it stood until 20 August 2026. They are
kept because §9 is only readable against them.*

### 4.1 The model must be asked before the image job starts

`mc_llm_sessions` takes the broker's workload lock for a whole run and waits
while `mc_broker.host_busy()` is true. A roll requested from inside a Forge
processing hook would therefore be an LLM run waiting for the image job that is
waiting for it. This was why the shape was *roll in a Gradio handler, apply in
the hook*, and why the browser had a bypass flag at all: the roll had to finish
before the native click was allowed through.

The constraint is real and has not gone away. What changed is that the hook now
*declares* that the image job is blocked waiting for it (`mc_broker.host_job()`),
which is the one case in which waiting for the image job is the wrong thing to
do. See §9.

### 4.2 The arming token was consumed, not checked

`consume()` took the token and cleared it in one locked step, so one roll
produced exactly one image. That mattered for more than tidiness: Stage 2's own
nested `process_images()` call, a queued request from a closed tab, and a second
Generate click during a batch would all otherwise have inherited permission
granted once.

The token is gone with the gap it guarded (§9). What replaced it is that there
is nowhere to put one: the hook that writes the prompt is the hook that applies
it, on one thread, in one call. Re-entrancy is the only part that still needs a
guard, and it is a flag on the script rather than a nonce.


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

That reasoning still holds and the fields it produced are still the fields. What
did **not** hold was the conclusion originally drawn beside it — that the Creative
keys should therefore be diagnostic and never pasteable. §10.5 is where that was
undone, and why leaving it as it was made an ordinary paste fail to reproduce the
image it came from.

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

- a task is claimed with ``modules.progress.add_task_to_queue`` and
  ``start_task``, which is what ``modules.call_queue`` does around every ordinary
  Gradio GPU call, and released with ``finish_task``;
- ``shared.state.textinfo`` carries the phase name and ``sampling_step`` /
  ``sampling_steps`` carry streamed characters, so **the existing** ``mc_progress``
  arithmetic describes the roll with no special case anywhere.

Since §9 there are two kinds of caller and the difference is one line. A roll
that runs *inside* a generation — which is every txt2img roll now — **borrows**
the bar the host already started for that generation: ``begin(claim=False)``, no
``start_task``, and above all no ``finish_task``, because finishing the image
job's task from inside ``before_process`` tells every poller that the generation
is over. A roll with no generation around it, such as LLM Studio's, **owns** its
task and does both. The counters are handed back either way; leaving them at a
finished roll's values would show the generation that follows as already
complete until its sampler overwrote them.

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
listening, and a button that does nothing is worse than no button. What happens
to the flag afterwards depends on whose bar it is, and the two answers are
opposite for the same reason. On an owned bar the roll is all that is running, so
the flag is cleared and a later press does not inherit a stop nobody meant for
it. On a borrowed bar the roll is the first part of a generation that is already
running, so Interrupt means *stop this generation* — the flag is left exactly as
the user set it and the host's own loop reads it a moment later.

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
set now, and ``TestTheBarNeverBlocksTheRunItDescribes`` drives a real roll with
the real ``host_busy`` — not the stub every other test in that file uses — and
asserts it answered False at every frame.

Note what this does *not* license, because §9 is easy to misread as a reversal of
it. The rule is that a progress indicator must never make the host look busy. The
exception §9 adds is that a caller which the host job is genuinely blocked
waiting for may skip the *wait*, having said so explicitly. Those are opposite
halves of the same statement: `host_busy()` must keep telling the truth, and one
caller is allowed to know that the truth does not apply to it.

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

`tests/test_krea_creative_js.py` runs the browser file under node against a
synthetic clock and a fake page, which is the only way to ask the questions that
matter there. A click is dispatched at the real listener list and the native
submission has to happen; no timer may be armed by a press or by an hour of
nobody touching anything; and the hidden roll button the old gate pressed has to
go unpressed. Four assertions read the file rather than run it — no
`setInterval`, no `preventDefault`, no `.click(`, comments excluded — because a
file with no timers today grows one the next time somebody wants to know when
the server has finished something.

`tests/test_krea_progress.py` covers the reporting, and is mostly about the bar
being *given back*. Since §9 it also covers the borrowed bar (the roll must not
start or finish the generation's task, and must hand the counters back) and the
deadlock itself: `TestTheRollRunsInsideTheGeneration` sets `shared.state.job` the
way `call_queue` does, does not stub `host_busy`, and drives `before_process` on
a thread with a deadline — because a deadlocked hook hangs a test run rather than
failing it, and the deadline is the assertion.


## 9. The browser stopped being load-bearing (20 August 2026)

Reported as: *Creative Mode generations do not complete if the browser window
loses focus.* Confirmed, and worse than reported — they did not complete if the
tab was closed either, and that was by construction rather than by accident.

### 9.1 What was actually happening

A press of Generate did not start a generation. `model_chain_creative_krea.js`
intercepted the click in the capture phase, called `preventDefault()`, pressed a
hidden Gradio button to run the roll, and polled a hidden textbox on a
`setInterval` until the server wrote `ready:<token>` into it. Only then did it
click Generate a second time, with a one-shot flag that let that click through.

Every part of that after the press lived in the page:

- **The poll.** `setInterval(…, 100)` is throttled to one tick a second in a
  hidden tab and to one a minute under Chromium's intensive throttling after
  five minutes hidden, and does not run at all in a frozen or closed one. So the
  image started late — or never — in exact proportion to how little the tab was
  being looked at.
- **The timeout.** `TOKEN_TIMEOUT_MS` was fifteen minutes of `Date.now()`, which
  is wall clock: a throttled poll could burn through it while the roll had long
  since finished, and the outcome was a resolved-false promise and silence.
- **The second click.** The thing that actually started the image was a
  `click()` from JavaScript. Close the tab and the roll's result sat armed on the
  server with nothing left to spend it.

None of this was a bug in the JavaScript. It was the JavaScript doing exactly
what it was written to do, and what it was written to do was be the only place
the *ordering* in §4.1 could be enforced.

### 9.2 The fix is one declaration

`mc_broker.host_job()`: a thread-local, re-entrant context manager that says
*this thread is the host's own job, for as long as this block runs*.
`mc_llm_sessions._Gpu.acquire` consults `mc_broker.inside_host_job()` and skips
the `host_busy()` wait when it is true.

That is not a loosening of the rule so much as the rule read carefully. The wait
exists so that an LLM turn does not compete with a running image job. A turn the
image job is itself blocked waiting for is not competing with it — it *is* the
job, in the part of the job that happens before any sampling. Nothing else is
relaxed: the workload lock is still taken, so two LLM turns still serialise, and
that wait terminates because the turn ahead is running rather than waiting on
anything this generation holds. The image side takes no broker lock at all
(`mc_broker`'s own comments explain why), so there is no cycle left to close.

The permission is declared, not inferred. Nothing tries to work out whether a
thread "looks like" a job thread; a caller that knows it is holding up the host
says so, for exactly as long as that is true, and one function reads it.

### 9.3 What that let happen

`ScriptKreaCreative.before_process` does the roll now. One press of Generate
starts an ordinary Forge generation, and everything Creative Mode does happens
inside it, on the thread the host is already running it on. Nothing after the
press touches a browser.

The ordering §7.5 depends on is preserved rather than sacrificed, which is worth
saying because it is the non-obvious half: `before_process` runs at the top of
`process_images`, *before* the checkpoint is (re)loaded, so the writer still
meets the card in the same state it used to and `image_reserve_bytes()` still
means what it meant.

Deletions that follow from it:

- the click gate, the hidden run button, the hidden token box, the `task:` /
  `ready:` / `failed:` channel and the nonce that made two identical outcomes
  distinguishable;
- the one-shot bypass flag, and the entire class of bug where a boolean is wrong
  in one direction if nothing generates and the other if everything generates
  twice;
- the arming token, `Creative.arm`, `consume`, `disarm` and `armed` — permission
  to cross a gap that no longer exists;
- `after_component`'s capture of the native prompt box. `p.prompt` is what the
  host is about to generate from, it is present whether the press came from the
  tab or the API, and it is still there when the tab is not.

`javascript/model_chain_creative_krea.js` is a third of what it was and does one
cosmetic thing: paint the armed class on Generate while Creative Mode is on. If
every line of it fails, a button goes unpainted.

### 9.4 What is worse, and why it is worth it

The panel no longer streams the roll's status into its own status line, because
there is no open Gradio event to stream down: the press became a native
generation rather than a handler with an output list. Two consequences:

- The phase is on the progress bar instead (§7.3), which is where a user is
  already looking and is arguably where it belonged.
- The **Last creative roll** drawer is filled by a button rather than
  automatically. It reads `Creative.last`, which outlives the page — so a roll
  made before the tab was closed can be inspected in the tab that opens after
  it, which the streamed version could not do at all.

Failure behaviour changed too, and deliberately. The gate refused the *click*
when a roll failed, so a non-Krea checkpoint or a dead `llama-server` produced no
image. The hook cannot refuse a generation that has already started, and should
not want to: every failure path now logs why and generates the prompt the user
typed. For the checkpoint guard in particular that is the better answer anyway —
a short prompt is exactly what an SD 1.5 checkpoint wanted.

### 9.5 Two guards that had to be added

- **Re-entrancy.** With the roll inside the hook, a nested `process_images()`
  that carried scripts would start a *second* language-model request while the
  first was still streaming. Stage 2 runs with `p.scripts` unset so this is belt
  and braces, but the cost of being wrong is no longer a duplicate image. One
  flag on the script instance.
- **Panel values reach the hook.** `ui()` returns every control now, not just
  the toggle, so `before_process` reads the slider the user just moved rather
  than the last value written to preferences. They are this script's own
  arguments and reach neither Model Chain's preset list nor its infotext, which
  is what §2's "a separate always-on script" was for. A caller with no panel
  behind it — the API — sends only the flag, and the saved settings answer.


## 10. The control surface stopped describing the implementation (21 August 2026)

Reported as four things, which turned out to be one thing and a bug.

The four: the panel renders every axis as a permanent row whether or not it is
doing anything; the factory defaults put nine of the ten axes on Vary; there is
no way to say "vary this, but never that"; and there are no named profiles. The
bug: pasting a Creative image back does not reproduce it.

The one thing the first four have in common is that the interface was a picture
of the data model. Ten axes exist, so ten rows were drawn; three modes exist, so
three radio buttons and a value dropdown were drawn per row, giving twenty
controls before the user had made a single decision — nine of which already said
Vary, which is nine art-direction decisions arriving from nowhere.

The rule the rebuild follows is the design intent's own sentence: **show the
decisions the user has made, not every decision the software knows how to make.**

### 10.1 Natural is absence, so it takes no space

Natural was always defined as *no line in the brief*. The panel now says the same
thing about itself: a Natural axis has no row, no dropdown and no space. Adding a
direction adds a line; returning it to Natural removes the line. The Active
directions list is therefore the shortest true description of what Creative Mode
is doing, and it is short precisely when the configuration is simple.

`AxisSetting.mode` defaults to `NATURAL` for the same reason at a different
level: an axis nobody has configured — one a later package adds, one an older
profile does not mention — has to fail neutral. Silence is a thing a user can see
the absence of; a silently varied axis is not.

### 10.2 The one file edited in the vendored package

`creativity/defaults.json` shipped with nine axes on Vary. That is a statement
about how the feature should *open*, which is what a defaults file is for, and it
is the one file in the package that describes this installation rather than the
vocabulary — so it is the one file edited, and
`CREATIVITY_LIBRARY_SOURCE.txt` records the edit, the reason, the new digest and
the digest as delivered. Everything else in the package is still byte-for-byte
what arrived.

`excluded_values` is added to the same file, and an absent key still reads as "no
exclusions", so a package update that does not know about the key costs nothing.

### 10.3 Exclusion is a modifier of Vary, never a fourth mode

The gap was real: a user could allow everything on an axis or pin exactly one
thing, with nothing in between. The tempting fix is a fourth mode, and it is
wrong — "never harsh noon" is a statement about *how* to vary, and making it a
mode would force somebody who wants two treatments gone to stop varying
altogether.

So `AxisSetting` grew `excluded_ids`, and `_choose_variant` removes them from the
pool *before* it weighs anything. Two consequences worth stating:

- **Exclusion is absolute, anti-repetition is a weight.** Anti-repetition falls
  back to the unpenalised weights rather than refuse to direct an axis (§5.6);
  exclusion never does. A user who said "never this" said it about every roll.
- **An axis whose whole pool is excluded is dropped before the draw, not during
  it.** Left in, it would consume one of the activation slots the Creativity
  position allows and then produce nothing — so excluding one small axis would
  quietly leave every *other* axis directed less often. It is skipped instead,
  with a note on the recipe, a line in the status area and a warning in the log.
  `CreativeRecipe.notes` exists for that: the two wrong answers here are both
  silent, and one of them is choosing the value the user forbade.

The panel shows exclusions as `gr.Dropdown(multiselect=True)` over the axis's own
variants, storing ids and displaying labels — ids are stable by the package's
contract, labels are display text a package update may rewrite.

### 10.4 One panel, two surfaces; one profile store, its own file

`mc_creative_panel.py` is built by txt2img and by LLM Studio's Krea tab. Two
implementations of a ten-axis editor would disagree within a release, and the
first thing they would disagree about is what a fresh install does.

Gradio 4 cannot create a component after the page is built, so every row and
every editor is built up front and hidden. That is not a compromise: `visible=
False` removes the element from the layout rather than making it transparent, so
"build them all, show one" is the same thing on screen as "create one on demand"
and much simpler than a component pool.

Every handler ends in `Panel.render()`, which returns an update for every
component the panel owns, computed from the stored settings. The alternative —
each handler updating what it believes it touched — is how a panel ends up
showing an exclusion list for an axis that is no longer varying. It costs one
wide outputs list per handler, which is a thing to read once rather than a class
of bug to find repeatedly. A test asserts `len(render()) == len(outputs())`,
because a positional list that disagrees by one puts every update after the
mismatch on the wrong control.

Every one of those handlers is wired to `input` rather than `change`. `change`
fires when the *server* sets a value, and each handler rewrites the very control
that fired it — so `change` would be a feedback loop that only terminates
because the value written back equals the value just read.

Profiles are their own file, `krea_creative_profiles.json`, in the WebUI data
directory, following `mc_presets` exactly (temp file, atomic replace, damaged
store reads as empty). Not `mc_llm_state.preferences()`, which holds *current*
settings and is rewritten every time a slider moves — a list of complete named
configurations living in that file would put everybody's saved work in the path
of every preference write.

The **Factory** profile is built rather than stored: it is read out of the
package's own `defaults.json`, so a package that ships different defaults ships a
different Factory with no code change, and it cannot be deleted or overwritten.
It is what makes "put it back" always answerable, including when the chosen
default has been deleted behind the panel's back — that falls back to Factory
with a sentence rather than refusing to build, because a Creative panel that will
not build is a Creative Mode nobody can turn on to fix.

Opening the panel shows a profile and never applies one: the dropdown names the
profile the live settings were last loaded from (`krea_creative_profile` in the
preferences file), falling back to the nominated default and then to Factory. The
alternative — reapplying the default on every page build — would silently discard
whatever the last tab adjusted, every time a tab was opened. *Reset to default*
is how somebody asks for that deliberately.

A profile does not carry the enabled toggle. A profile says how the feature
behaves; whether it runs is a decision made at the moment somebody presses
Generate, and a preset that could flip it would be a preset that changes what the
button does.

### 10.5 The paste bug, and the two questions it conflated

`mc_infotext` used to say, in a docstring, that the Creative keys were diagnostic
and deliberately not pasteable: a pasted source prompt would either overwrite
what somebody was iterating on or silently re-run a language model. Both halves
of that are true. The conclusion was still wrong, because it answered a question
nobody was asking and left the real one unanswered.

Creative Mode assigns the expanded prompt to `p.prompt` *before* Forge records
infotext. The recorded `Prompt:` line is therefore the paragraph the image model
actually saw — which means a paste already restores everything needed to
reproduce the image, and the only thing standing in the way is that Creative Mode
is still on and will expand that paragraph a second time. The result is a picture
of the prompt of the picture.

So there are two questions and they now have two answers:

- **Reproduce the image.** One paste field, on the enabled checkbox, answering
  `False` for any infotext carrying `Krea Creative Mode`. Nothing else changes:
  the prompt, seed, checkpoint, sampler and size are the host's and always were.
  The status line says what happened. An infotext with no Creative keys returns
  `None`, which is how the host is told to leave a control alone — an ordinary
  image must not be able to switch a feature off any more than on.
- **Restore the workflow.** A separate button, under *Continue from a pasted
  image*. It is the only handler in this extension that writes to a native
  control, and it writes to exactly one: the txt2img prompt box, grabbed by its
  own `elem_id` in `after_component`, because a recorded *source phrase* has
  nowhere else to go. A test asserts that no other handler on the panel names a
  component outside it.

  It is also the only thing in the extension that switches Creative Mode *on*,
  and that is deliberate rather than convenient: the paste switched it off so the
  picture would reproduce, and this button is the request to do the opposite. A
  short idea generated with the writer switched off is not a smaller version of
  the feature — it is a bare phrase handed to Krea 2. A test counts the
  occurrences and fails at two.

The capture happens on every paste (piggy-backed on a paste field, because a
paste field is the only callback the host offers) and is stashed in
`mc_creative_krea.pasted`. It holds a parsed record and nothing that could reach
a generation on its own.

**Exact replay.** Re-rolling at the recorded Creative seed is *not* a
reproduction: the draw is weighted by recent history, and a machine six months
later has different recent history. `director.replay()` rebuilds the recipe from
the recorded ids instead — nothing drawn, no history consulted, ids the current
package no longer has dropped with a note rather than substituted. It is armed
explicitly, for one generation, and says so on the panel while it is armed.

That is an arming mechanism, and §1 records that the last one was removed on
purpose, so the difference is worth stating. The old token held a **finished
prompt** — a model's output, made before the click, waiting for a generation to
spend it, which is how a closed tab used to strand work. `ReplayPlan` holds a
short list of variant ids the user chose, that they can read before pressing
anything, that no model produced, and that reaches nothing on its own. The roll
still happens inside the generation and the writer is still called exactly once;
all that changes is where that roll's art direction comes from. It lives beside
the session rather than on it, it is taken before the roll is attempted so a
failed roll cannot leave it armed, and a replay is not written into the recent
memory — its ids were recorded when they were first drawn, and writing them again
would push a user's own reproduction away from what they asked to reproduce.

### 10.6 What the panel argument list looks like now

Three controls per axis rather than two — mode, pinned value, exclusions — in the
library's own axis order, parsed in exactly one place per surface
(`mc_creative_panel.axes_from`). A tuple of the wrong length is refused outright
rather than unpacked as far as it goes: reading three controls per axis out of
two produces a *valid* configuration nobody chose, which the caller would then
save over the one they did. Refusing means the saved settings answer, which is
the same thing an API request with no panel behind it gets.


## 11. What the reading step actually costs (21 August 2026)

Reported as *"why does the reading input prompt step take over ten seconds, on
both CPU and mixed mode? I want it to start creating output fast."* The
`llama-server.log` answers it precisely, and the answer changed two things in
the code.

### 11.1 The cache is working; the brief is the cost

A run mid-log, on a warm server:

```
prompt eval time = 10737.63 ms / 382 tokens (28.11 ms per token)
stop processing: n_tokens = 1028
```

1,028 prompt tokens, 382 evaluated, **646 reused**. The 646 are Krea's
instruction, which is the same bytes on every roll and which llama.cpp's prompt
cache therefore answers for. The 382 are the user's line and the creative brief.

So the reading step is not the request being slow. It is the brief being read,
and the brief cannot be cached *by construction*: Vary means a different one
every roll. Across the same log the relationship is exactly linear — 92 tokens
at Creativity 5 with one direction, 520 at Creativity 10 with six, at a flat
26–36 ms per token.

Two rows of that log are the whole answer to "how do I make it fast", though:

```
prefill  0.6s for 482 tokens (843.6 tok/s)     ← weights on the card
prefill 10.9s for 479 tokens ( 43.7 tok/s)     ← weights in system RAM
```

Same machine, same model, twenty times the speed. Mixed placement is defined as
*no resident layers* (§5, `Config.__post_init__`), so its prefill runs on the
processor exactly as CPU placement's does. Nothing in the prompt is worth
optimising next to that.

### 11.2 The bar was pricing bytes that cost nothing

`_prompt_size()` counted Krea's instruction, the user's line and the brief —
everything the request *carries*. What llama.cpp *evaluates* on a warm server is
the last two. Counting the instruction anyway made the read phase over-predict a
short brief, and worse, made the learned `krea:read` rate drift with the mix:
the same seconds were being divided by a character count that included a
constant, so the seconds-per-character it learned depended on how big that
roll's brief happened to be next to the two kilobytes of instruction.

It now counts the instruction only when the server is cold, where it really is
read. `krea:read` means one thing again, and both the bar and §11.3 use it.

### 11.3 The panel says what a direction costs

The progress bar is the first place a user sees the cost and the last place they
can do anything about it. Two of the three levers — how many directions, and the
Creativity position — are decisions made in the Creative Controls drawer, so the
drawer now carries the number:

> *About 1,299 characters of brief — roughly 4s of reading before the writing
> starts with the model in system RAM. The brief is different every roll, so it
> is the one part of the request that can never come out of the model's cache.*

The characters are a three-seed average of the brief this configuration produces
(at mid Creativity the activation count is itself a draw, so one sample is one
possibility rather than the typical one). The seconds are `mc_progress.measured
("krea:read")` — this machine's own measurement, out of the same store the bar
predicts from, so the panel and the bar cannot disagree. The placement clause is
read from the runtime configuration, because the lever it names is the one this
panel cannot pull.


## 12. A slider describing a scale it had nothing to apply (21 August 2026)

Reported as *"the creativity mode slider is not working — even at 0 creativity, a
one word input prompt leads to a nearly 400 character output"*, alongside a copy
of the repository from before the rebuild.

Both halves of that are working exactly as designed, and both looked broken for
the same reason: the panel was describing what the *scale* means without saying
whether there was anything to apply it to.

### 12.1 The slider was telling the truth about the wrong thing

```
Creativity 10, every axis Natural
  → 0 axes directed, brief 0 characters
  → panel said: "Creativity 10 — extreme direction on every eligible axis"
```

`variation.describe()` names the position on the scale, which is half an answer.
The other half is the axis configuration, and after §10 the shipped one is every
axis Natural — so a fresh installation's slider genuinely does nothing to the
brief at any position, and said the opposite at every position.

`mc_creative_panel.describe_creativity()` now answers with both halves, and names
the *two* ways a direction produces nothing, because they are indistinguishable
from outside: no directions at all, and directions that exist but sit below
Creativity 2, where the Director emits nothing by design and by promise. The
status line the toggle writes carries the same information, because while the
drawer is shut it is the only Creative text on screen and "Creative Mode is on"
is deeply misleading on a configuration that directs nothing.

### 12.2 The 400 characters are the writer, not the direction

Expanding a short idea into a full Krea prompt is what the writer is *for*; art
direction changes what it expands into, not whether it expands. A one-word
prompt at Creativity 0 with no directions is the plain Krea expansion, which is
also exactly what Creativity 1 is defined as (§5) — and the pre-rebuild copy the
user attached confirms it: `variation.py`, `enhancer.py`, `expansion.txt` and
`mc_llm_sessions.py` are byte-identical across the rebuild. Nothing about how
the prompt is written changed.

What changed is the default configuration, and with it the *feel*: nine axes on
Vary produced a richer expansion than none. That is the trade §10.2 made
deliberately, and the honest way to give it back is one click rather than a
different default — hence the second built-in profile, **Everything varies**.

### 12.3 The other twenty seconds

The same log:

```
read    4.2s for 116 tokens   (28 tok/s)   ← the brief, after §11
write  21.3s for 103 tokens   (4.8 tok/s)  ← the expansion itself
```

§11 moved the reading half from ten seconds to four by pricing only what is
actually evaluated, and the neutral defaults cut the brief that feeds it. The
remaining twenty seconds is the model writing four hundred characters at five
tokens a second, which is a 12B in system RAM and is not something any control
on this panel changes.

So the cost line names both halves — *"roughly 4s of reading, then about 20s of
writing"* — from the same measured store, because a user told about four seconds
who then waits twenty-five is owed the other twenty-one.


## 13. Why the card was idle and the writer was slow (21 August 2026)

Reported as *"you still didn't explain what went wrong"*, with four runs — cold
CPU, warm CPU, cold Mixed, warm Mixed — and two observations: five seconds
before the first character on a warm server, and Mixed no faster than CPU.

Both are explained by the same two lines of the user's own console log, and
neither had anything on screen saying so.

### 13.1 "Mixed" was doing no GPU work, and said the opposite

```
Model Chain: starting llama-server — Q4_K_M on NVIDIA GeForce RTX 3090,
             system RAM (no GPU offload), 8,192 token context
```

Mixed placement is recorded with `gpu_layers = "0"`, `Config.__post_init__`
enforces it, and `_layers_argument` emits `--n-gpu-layers 0`. llama.cpp with no
offloaded layers runs every matrix multiply on the processor. So Mixed and CPU
are the same computation — 4.2 against 5.3 tokens a second in the user's four
runs, which is noise — and the card is named on `--device` purely so it is
visible.

The vendored device describer said, in the Setup list the user chose from:

> mixed: model in system RAM, **card used for processing**

which is how somebody ends up asking why the card is not being used. The
vendored file stays byte-identical (`prompt_master/VENDORED_FROM.txt`);
`mc_llm_setup.describe_device` overrides the wording for the mixed case, and
`_warn_about_an_idle_card` says the same thing in the log at every start where a
card is present and no layers are offloaded.

**What it is not**, and this is the part worth keeping: a bug to be fixed by
offloading anyway. The same log says

```
Model Chain: the image generation that follows a Krea roll is short 9.6 GB
             and nothing evictable was found; expect the driver to spill
```

The Krea 2 stack alone wants roughly 17.6 GB of a 24 GB card and is already
spilling. There is no room to give the writer, and taking some would slow the
image down to speed the prompt up. Mixed is doing the right thing here. It was
only describing itself wrongly.

### 13.2 The five seconds are prefill, and llama.cpp says how much

From the same run's `llama-server.log`:

```
restored context checkpoint (pos_max = 460, n_tokens = 461, size = 76.539 MiB)
prompt eval time = 5565.34 ms / 114 tokens (48.82 ms per token, 20.48 tok/s)
```

The prompt cache restores to a *checkpoint*, at token 460, and everything after
it is evaluated again: the tail of Krea's instruction plus the whole user turn.
That is 66–118 tokens per warm roll, and on a processor a small batch is much
worse per token than a large one — 55 ms/token on 19 tokens against 21 ms/token
on 574. Five seconds of arithmetic, not five seconds of overhead. There is
nothing between the press and llama.cpp to remove.

### 13.3 The measurement nobody was keeping

`Runtime.speed_note()` has always read llama.cpp's own tokens-per-second out of
its log after each request — and printed it into a log line and thrown it away,
while `mc_progress` estimated the same quantities from character counts.

Those numbers are now kept, per backbone (`llm:write:<id>`), because that is the
axis they differ on most and the difference is the opposite of what the
catalogue's sizes suggest:

```
Gemma 4 12B QAT   dense, ~7.4 GB     4.9 tokens/s     ← "Recommended"
Gemma 4 26B-A4B   MoE,  ~16.8 GB    12.8 tokens/s     ← 2.6× faster
```

Generation from system RAM is bandwidth-bound, so it follows the *active*
parameters per token: a mixture-of-experts touches about 4B of its weights,
a dense 12B touches all twelve. The user had switched from the second to the
first — from the entry labelled *Current large baseline* to the one labelled
*Recommended* — and got two and a half times slower, with nothing on screen
saying that was the trade.

`ManagedModel.describe()` now carries `measured here: 12.8 tokens/s` beside the
size, and the progress bar predicts from the running backbone's own key rather
than a rate shared across all of them (which meant every switch spent its next
several rolls predicting the previous model's speed, and the Creative panel
quoting it).

### 13.4 What there is no lever for

No CPU equivalent of sage attention exists, and flash attention is a CUDA kernel
that does nothing when no layers are on the card. The writer's output length is
the product — it is a Krea prompt, and a truncated one is not a faster prompt,
it is a broken one. What is left is the three things above: which backbone,
where it runs, and how much brief it is asked to read.


## 14. Spatial BBOX: composition the writer does not own (21 August 2026)

Companion to the *Krea 2 Creative Mode Spatial BBOX* design intent v0.1
(21 August 2026). Section numbers in this part are that document's.

The feature in one sentence: the user draws boxes, types a prompt into each one,
and the boxes reach Krea 2 as a structured prompt whose *scene* Creative Mode
wrote and whose *elements* nothing wrote — they are what was drawn.

### 14.1 What was built

| File | What it is |
| --- | --- |
| `prompt_master/krea/spatial.py` | the pure half: parsing, validation, the deterministic hints, the compositor |
| `prompt_master/krea/composer.py` | pass 2's instruction, and the strict reader for what comes back |
| `mc_spatial.py` | the host glue: preferences, the pass-2 run, and what one spatial generation records |
| `mc_llm_sessions.krea_compose` | one Composer request, beside `krea` and unlike it in three ways |
| `mc_llm_progress.Pass` | which of the two passes the bar is describing, and what it learns from it |
| `scripts/model_chain_krea_creative.py` | the panel block, the editor markup, and the four steps after the roll |
| `javascript/model_chain_spatial_krea.js` | the editor: drawing, dragging, z-order, serialization |
| `mc_infotext.py` | ten `Krea Spatial *` keys, and the second paste field that switches the feature off |
| `tests/test_krea_spatial.py` | coordinates, the compositor, isolation, failure, infotext |
| `tests/test_krea_spatial_js.py` | the editor run under node against a fake page built from the real markup |

### 14.2 The division that the whole thing rests on

**A model may write text fields. Code supplies the structure.** §2.3 states it and
this implementation takes it literally: `spatial.compose()` builds the elements
array from validated user state, and the only two strings it will accept from
anywhere else are `high_level_description` and `background`.

That is not a preference about tidiness. Asking a model for the finished JSON
fails in five ways and every one of them is silent — a coordinate drifts forty
units, two regions merge, a user's wording comes back "improved", the reply is
almost-valid JSON, and the same layout produces different bytes tomorrow so a
Smart-against-Direct comparison compares two draws. The adversarial test is the
short version: pass 2 is handed a reply carrying `elements`, `regions`, `bbox`
*and* a rewritten region prompt, and the finished prompt is unaffected — because
only two names are ever read out of it.

### 14.3 Why the Composer does not carry Krea's instruction, and what that costs

This is the one place where the fast thing and the right thing point in opposite
directions, so it is worth the paragraph.

llama.cpp resumes a prompt at its common prefix with the *previous* one (§11.1,
§26.2 of `07-llm-studio.md`). Krea's expansion instruction is ~460 tokens, it is
identical on every roll, it is prefilled at startup, and it is the reason a warm
roll reads only the creative brief. A second pass that reuses that system
message keeps the prefix; one with its own system message pays for its own
instruction *and* makes the next roll pay for Krea's again.

On a processor-only placement at ~30 tokens/second that is about fifteen seconds
each way. So the tempting design is `expansion.txt` plus a composer addendum,
exactly the way `enhancer.system_prompt` appends the reference rules.

It is the wrong design, and `expansion.txt` says why in its own words:

> 1. **Faithfulness First:** … expand …
> 6. **Structure:** Write one cohesive paragraph after the thinking block. No
>    bullets, JSON, or markdown.

The Composer's job is to *shorten* and to return two labelled fields. Putting an
addendum under those two rules would be this repository telling a model to
disregard half of a vendored file, inside the context window, on every spatial
generation — and betting the pass's reliability on a 12B model resolving the
contradiction the way we meant.

So the Composer has its own short instruction and the thirty seconds are spent.
Three things make that an acceptable trade rather than a regression:

- **It is opt-in twice.** Smart mode *and* at least one region. Smart with an
  empty canvas makes no request, so a fresh install pays nothing.
- **Direct is one radio button away** and makes no second request at all. It is
  the A/B control §5.5 asks for and it is also the answer for anybody on a
  processor who wants their thirty seconds back.
- **On a card it is about half a second each way.** The cost is a property of
  running the model in system RAM, which is the same lever §13.4 already names.

Nothing about pass 1 changed. With Spatial Layout off, the writer is sent the
same bytes it was sent before this feature existed — checked as bytes, in
`test_with_no_layout_the_turn_is_the_turn_it_always_was`, because "the same
prompt-cache prefix" is not a thing that can be argued from a diff.

### 14.4 The one sentence pass 1 is told

§2.2 says the Creative Writer should not receive individual region prompts, and
may receive a generic note about placement. `spatial.PLACEMENT_NOTE` is that
note and it is the whole of what pass 1 learns about the layout.

The reason is worth stating as a mechanism rather than a rule. Handing pass 1 the
region text would put the user's own words through a rewriter; the compositor
would then place the *original* words in the elements array beside the rewrite in
the scene; and the image model would be asked for two of the same subject. The
duplication §9.8 wants reduced would be introduced by the very pass meant to
avoid it.

The note is appended only when a region survived validation, which is what keeps
the Creativity-1 compatibility guarantee (§3) intact: at Creativity 1 with no
layout the user turn is still byte-identical to the legacy one.

### 14.5 The hints are five decisions, and all five are code's

`desc` is the raw region prompt, verbatim and first, then framing, then angle,
then position, then size. Every one of the last four comes out of a fixed
vocabulary in `spatial.py`, and the fixed vocabulary is also what the editor's
dropdowns are built from — `spatial_editor()` reads `FRAMINGS` and `ANGLES`
rather than restating them — so a framing that can be chosen is a framing that
renders a phrase. A selection this build does not know is dropped *with a note*
rather than passed through, because a silently discarded choice is the failure
§10 spends a section on.

The position hint uses the same nine cells as the thirds grid the canvas draws.
That is deliberate: the sentence the model reads and the guides the user drew
against describe the same division of the frame.

Sizes are five named bands rather than a percentage. "Occupying 8.54% of the
frame" is a number a text-to-image model has no calibration for.

### 14.6 Coordinates, and the one thing normalization does not solve

0..1000 on both axes, clamped, ordered, and a zero-area box refused outright.
Clamping and ordering are recoveries — a drag past the edge meant the edge, a
drag from bottom-right is the same rectangle — and a zero-area box is not a
recovery, it is a click, so it is skipped with a reason. §10's "never guess
replacement coordinates" is enforced by there being no code that could.

Because the coordinates are fractions of the frame, a resolution change at the
same aspect ratio is not a change at all, and the test says so by composing the
same layout at 1024×1344 and 1536×2016 and asserting the bytes are equal.

An **aspect** change is a different matter and §6.4 asks for reprojection plus a
non-blocking warning. What is implemented is the warning, and the boxes are left
exactly where they are. Reprojection would mean this code deciding which of
somebody's boxes deserved to keep its shape and which deserved to keep its
position, and there is no answer to that which is right for both a face and a
horizon. The warning is a line of text in the editor's header — nothing to
dismiss, nothing that can be dismissed by accident, and no path on which layout
state is lost.

`aspect_ratio()` reports the exact reduced ratio when it is small enough to
recognise and snaps to a common one when it is not: 832×1216 reduces to 13:19,
which is true and useless, and is 2:3 to within two and a half per cent.

### 14.7 Two switches, and why the paste turns both off

A spatial image's recorded `Prompt:` line is the finished structured prompt —
aspect ratio, scene, background and every element. So an ordinary paste
reproduces it exactly, the same way §10.5 made an ordinary paste reproduce a
Creative image, and for the same reason: the prompt was assigned before Forge
wrote the infotext.

What that requires is that *both* features come back off. Creative Mode off with
Spatial Layout on would compose boxes around a prompt nobody expanded; Spatial
on with Creative off would build a BBOX prompt whose `high_level_description` is
an entire BBOX prompt. Two paste fields, two `False`s, and `None` from both for
an image that carries neither key — because an ordinary image must not be able
to switch a feature off any more than on.

The Spatial toggle exists partly for this. §6.1's sketch shows a region count and
an Edit button with no checkbox, and an explicit toggle is one more control than
that — but it is what makes "off" a thing a paste can restore and a thing a test
can assert, and it is the same shape as the Creative toggle above it.

### 14.8 What is recorded, and the one key that is deliberately conditional

Ten keys, all namespaced `Krea Spatial *`. The layout is the whole editable
state, re-serialized from what parsed rather than passed through — so what an
image records is what this build actually used, with coordinates clamped,
ordering canonical and unknown fields gone.

`Krea Enhanced Scene` and `Krea Spatial Scene` are written in **Smart mode only**,
and that is §5.2's argument applied one layer out rather than a different one. In
Direct mode the enhanced scene *is* the `high_level_description` in the Prompt
line, and a second copy would be several hundred bytes repeating what the file
already says. In Smart mode it is genuinely nowhere else — the Prompt carries the
reconciled scene — and without it the A/B comparison §11 asks for cannot be done
after the fact. There is a checkbox for people who would rather have the bytes
back.

### 14.9 The editor, and the hidden textbox that is not a channel

`javascript/model_chain_spatial_krea.js` owns interaction and serialization; the
markup is built in Python so the vocabularies have one source; Python owns
validation, both models, composition, infotext and replay. That is §6.5.

The state box is the part that deserves suspicion, because it is the same shape
as the thing §9 deleted: a hidden Gradio textbox that JavaScript writes into.
The difference is entirely in what waits for it. The old one was *polled* — an
image could not start until a `setInterval` in the page saw a token appear in it,
which is why a hidden tab made a generation late and a closed tab made it never
happen. This one is an input, read with the slider and the checkbox when Generate
is pressed. Nothing polls it, nothing is armed, and no generation is held up by
it.

That claim is checked rather than asserted. `tests/test_krea_spatial_js.py`
drives the real file against a synthetic clock and a fake page: it runs an hour
forward with nobody touching anything and asserts no timer exists, asserts the
file never names `txt2img_generate`, and asserts it registers no listener on it.
Two assertions read the file rather than run it — no `setInterval`, no
`setTimeout` — because a file with no timers today grows one the next time
somebody wants to know when the server has finished something.

The fake page is built from `spatial_editor()`'s own markup, so a control renamed
in Python and not in JavaScript fails in a test run rather than as a dead button
in somebody's browser.

The overlay is moved to `document.body` on first wire. A `position: fixed` modal
inside a Gradio accordion is one `overflow: hidden` or one `transform` away from
being invisible, and neither of those is this extension's to promise about a
theme.

### 14.10 Failure, in the order it is tried

Every one of these was already in §10 and every one of them falls back
*backwards* — to the thing that was true one step earlier — and says so:

| What failed | What happens | Where it is said |
| --- | --- | --- |
| No regions survived validation | an ordinary Creative Mode generation | log, and the panel's summary line |
| Spatial Composer (timeout, empty, not the schema, interrupted) | Direct merge: pass 1's scene, the same boxes | log, and a line on the result |
| The layout could not be read | nothing is applied, and nothing is overwritten | log, panel, and a line on the result |
| The compositor raised | the writer's paragraph, unchanged | log, and a line on the result |
| The writer failed | the existing Creative Mode fallback: the typed prompt | the existing sentence on the result |

The last row is the one that is arguable, and it is the design intent's answer
rather than this implementation's preference. §10 says a Creative Writer failure
uses *the extension's existing Creative Mode fallback behaviour*, which is to
generate the prompt exactly as typed. Composing boxes around an unexpanded phrase
would be a third behaviour nobody specified, so the layout is reported as not
applied and the press that follows tries again.

Interrupt during the Composer pass is worth naming separately: it skips the
Composer pass and does not cancel the image, because by then the image job is
already running and pass 1 has already succeeded. The host's own loop reads the
same flag a moment later and stops the sampling if that is what was meant.

### 14.11 The bar describes two passes now

`mc_llm_progress.Pass` names which one. The writer keeps `krea:read`,
`krea:write` and `krea:reply`; the Composer measures under
`krea:compose:*` and says *Reading the layout* and *Reconciling the scene with
the layout* instead.

Separate keys because the two are unalike in exactly the way the store models:
the writer reads a creative brief and answers with a Krea paragraph, the Composer
reads a scene and a layout and answers with two short fields. One shared
`krea:reply` would learn the average of the two and then predict neither — which
is the same mistake §13.3 fixed between backbones, one level down.

The Composer borrows the generation's bar exactly as the roll does (§7.3): no
`start_task`, no `finish_task`, counters handed back on every path out.
