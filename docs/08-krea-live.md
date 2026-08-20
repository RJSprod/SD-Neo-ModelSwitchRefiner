# Krea Live and the Creativity control — implementation notes

Companion to the *Krea 2 Live Mode + Shared Creativity Control* design intent
(revised proposal, 20 August 2026). That document states what the feature is;
this one records the choices made against it, the places it was followed
differently, and the two or three things that will otherwise be rediscovered the
hard way.

Section numbers below are the design intent's.


## 1. What was built

| File | What it is |
| --- | --- |
| `prompt_master/krea/variation.py` | the 0–10 scale, as one table and one function |
| `prompt_master/inference/llama_client.py` | a whitelist for optional sampler fields |
| `mc_llm_sessions.py` | the Krea writer takes a Creativity position |
| `mc_llm_krea_panel.py` | the manual slider |
| `mc_live_krea.py` | the Live session: cache, arming token, one call |
| `scripts/model_chain_krea_live.py` | the txt2img strip and the processing hook |
| `javascript/model_chain_live_krea.js` | debounce, the Generate gate, the menu |
| `tests/test_krea_live.py` | the scale, the one-call rule, the hook |
| `tests/test_krea_live_js.py` | the debounce and the bypass, executed under node |

The rollout order in §9 was followed, and all four phases are in.


## 2. Where the design intent was followed exactly

- **One LLM call per new prompt-authoring state** (§2.1). The cache key is
  `mc_live_krea.cache_key`, over exactly the four inputs the writer sees —
  source text, Creativity, prompt-seed policy or value, and the identity of the
  loaded LLM. Nothing else is in it, and the exclusion is asserted directly
  rather than left to be inferred from the call sites.
- **No candidate generation** (§5.5). `SamplingProfile` has no `candidates`,
  `candidate_count`, `judge` or `novelty_score` field, and a test asserts their
  absence rather than merely their non-use.
- **Creativity 1 is exactly today's configuration** (§5.2). Temperature 0.6,
  top_p 0.9, and the new sampler fields *omitted from the payload*, not sent
  with guessed neutral values. `variation.legacy_profile()` exists so that the
  promise has a name a test can assert against.
- **Krea's instruction is untouched** (§2.2). `variation.py` cannot read a file,
  cannot import the enhancer, and contains no prompt text; a test parses its AST
  to say so, which survives the docstrings that talk *about* the instruction.
- **Forge remains the image backend** (§2.4). The only thing the processing hook
  does is assign `p.prompt` and update `p.extra_generation_params`.
- **The visible prompt stays the user's** (§2.5, §6.7). No handler anywhere has
  the txt2img prompt component as an output, and a test checks that.
- **Steps has one source of truth** (§6.3, §8.8). The strip's box and the native
  slider are bound to each other on `input` and `release` — user-input events —
  so the two never answer each other in a loop, and there is no third value.
- **Interception happens before image generation** (§6.6). The gate is a
  capture-phase click listener; the hook cannot start inference and does not try.


## 3. Where it was followed differently, and why

### 3.1 The expanded prompt is not written into infotext twice

§6.7 lists four values to record: source prompt, expanded prompt, Creativity and
prompt seed. Three of them are recorded. The expanded prompt is not, because it
is already in the infotext as the image's own `Prompt:` line — that is what Live
substituted before sampling started. A second copy under a `Krea Live Expanded
Prompt` key would add a few hundred bytes to every PNG to repeat what the file
already says, and would create a second copy that a later paste could disagree
with.

What is recorded instead is `Krea Live LoRAs`: the pinned tags, which are the
only difference between the writer's paragraph and the prompt that was generated
from. With the source, the position, the prompt seed and the tags written down,
the expansion is recoverable from the prompt line exactly and the run repeats.

### 3.2 Live is text-only, and says so on the page

§8.3 asks for this distinction to be made explicit in implementation
documentation, and offers text-only as the primary path for the base feature.
It is the only path here. A reference image needs a captioning pass before a
prompt can be written about it, and that is one model request per reference on
top of the writer's — which is a different product from "one request per prompt",
not a larger version of it. The Live configuration area says so where somebody
would otherwise go looking for the upload slots, and points at LLM Studio →
Krea 2, which has them.

### 3.3 The Live strip is its own always-on script

§8.7 offers either `scripts/model_chain.py` or a small separate script.
Separate, for three reasons. Model Chain's `ui()` returns a long argument list
that travels in presets and infotexts, and Live's controls have no business in
either. Model Chain is a large, long-settled thing and a Live gate that failed
to build must not be able to take the two-stage chain down with it. And Live is
txt2img-only for a different reason than Model Chain is, so the two `show()`
methods happen to agree today and are not the same decision.

The one thing it *does* return as a script argument is the enable checkbox,
which the hook reads: an expansion armed and then disarmed before the click
landed is not generated from.

### 3.4 The one edit to the vendored tree

`prompt_master/` is a byte-identical copy of another repository's package, and
`VENDORED_FROM.txt` says so and says not to hand-edit it. §8.2 asks for
`inference/llama_client.py` to be modified anyway, and it was — so the edit is
recorded in that file, in `docs/07-llm-studio.md`, and here, rather than left
for a `diff -r` to find.

It is deliberately the smallest edit that could work: one optional argument
defaulting to `None`, and a whitelist that returns an empty dict for `None`, so
the payload built for every pre-existing caller is unchanged byte for byte. The
whitelist is in the client rather than in a wrapper because it is a statement
about what *that client's* request may contain — a filter applied on the way in
by an `mc_llm_*` module could be bypassed by anything holding the client, and
`mc_llm_sessions` holds it.

### 3.5 Dynamic temperature and XTC are whitelisted but unused

§5.4 names `top_k`, `min_p`, dynamic temperature and XTC as examples of controls
that *may* be progressively relaxed. Two are used. The client's whitelist
accepts all four plus six more, so adding a row to the table is the whole change
when there is a reason — but a sampler setting shipped on the strength of
sounding adventurous is a setting that makes value 9 produce broken sentences,
and neither of the two left out has been measured against real Krea prompts on
real models here.

### 3.6 Random image seeds are left entirely to Forge

§7.7 asks that a random image seed be resolved to a concrete value before
submission so that rerolls can be shown to have used different diffusion seeds.
Forge already does exactly this: `-1` is resolved per generation and the
concrete value is written into infotext as `Seed:`. Re-implementing it would put
a second seed resolver in front of the host's, which is the kind of duplicate
semantics §6.3 rules out for Steps. Nothing was added.

### 3.7 Generate forever is refused, not routed

§7.8 offers two options and recommends the second for a first implementation.
Refusal it is. The browser gate can tell a synthetic click from a real one
(`event.isTrusted`), so a native repeat loop is stopped through the host's own
`generateOnRepeatInterval` and a line appears in the Live status saying to use
Reroll instead. Two repeat schedulers racing to press the same button is not
something worth refereeing.


## 4. Two things worth knowing before changing this

### 4.1 The hook must never ask for an expansion

`mc_llm_sessions` takes the broker's workload lock for a whole run and waits
while `mc_broker.host_busy()` is true. An expansion requested from inside a Forge
processing hook would therefore be an LLM run waiting for the image job that is
waiting for it. This is why the shape is *request in a Gradio handler, apply in
the hook*, and why `Live.consume()` is deliberately the entire vocabulary the
hook has. It is also why the browser has a bypass flag at all: the expansion has
to finish before the native click is allowed through.

### 4.2 The arming token is consumed, not checked

`consume()` takes the token and clears it in one locked step, so one expansion
can produce exactly one image. That matters for more than tidiness: Stage 2's
own nested `process_images()` call, a queued request from a closed tab, and a
second Generate click during a batch would all otherwise inherit permission that
was granted once.


## 5. Tests

`tests/test_krea_live.py` is mostly a request counter. `FakeClient.calls` gets
one entry per completion llama.cpp was asked for, with the sampling that was
asked for in it, and the assertions are about that number: one for a new source,
zero for a reroll, zero for a LoRA weight change, one for a moved slider. The
compatibility anchor is checked as a payload rather than as a prompt, because
that is what it is.

`tests/test_krea_live_js.py` runs the browser controller under node against a
synthetic clock. The debounce and the one-shot bypass are state machines, and
reading them proves nothing: a fake `setTimeout` and a counter prove that typing
does not generate, that each edit restarts the timer, that the delay is the one
showing on the strip, and that the programmatic click made after an expansion
spends the flag on the way past and exactly one native generation happens. It
skips where node is absent, as `tests/test_llm_studio_js.py` already does.
