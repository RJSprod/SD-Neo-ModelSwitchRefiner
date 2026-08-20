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
| `tests/test_krea_creative.py` | the library, the Director, the scale, one-call |
| `tests/test_krea_creative_js.py` | the bypass and the absence of a scheduler |


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


## 7. Tests

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
