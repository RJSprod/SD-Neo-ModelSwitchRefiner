# Composable generation memory and persistent LLM residency

Design intent: *Composable Generation Memory & Persistent LLM Residency*.
Implementation: `mc_plan.py`, `mc_plan_panel.py`, and changes to
`mc_llm_runtime.py`, `mc_memory.py`, `mc_creative_krea.py` and both txt2img
scripts.


## 1. The problem, in the words of a log

Every part of this extension sized memory against *now*: how much was free at
this instant, what the loaded checkpoint weighed, what the running server held.
That is the right question for one workload and the wrong question for a
generation assembled from several, because the phases do not all happen at once
and the first one to ask gets an answer computed as though it were the only one.

The Creative Writer is the first one to ask, and it runs **before the checkpoint
is loaded**. What is free at that moment is very nearly the whole card. What is
free three hundred milliseconds later, when Stage 1 loads and Stage 2 follows
it, is not.

From one user's `llama-server.log`, a single session:

| Symptom | Count |
| --- | --- |
| llama-server starts | 71 |
| starts that died loading the model | 47 |
| `cudaMalloc failed: out of memory` | 24 |
| starts that died at argument parsing with `invalid device: CUDA0` | 31 |

and the negotiated context stepping `7168 → 8192 → 7168` across consecutive
generations, because free VRAM stepped with it. Every one of those steps changes
the placement signature, which is a restart, which loses llama.cpp's prompt
cache: about thirteen seconds of prompt evaluation paid again, twice per
generation when Smart Spatial is on.

The two hard failures are the same story further along. Five consecutive
generations had all three start attempts fail with `cudaMalloc failed: out of
memory` on a card whose own `device_info` line, printed seconds earlier in the
same process, said 22.7 GB free — llama.cpp read the free figure, the image side
took the card during the load, and the language model lost the race. And a run
of 31 starts never reached the model at all: a CUDA context needs VRAM of its
own before any device can be enumerated, so a process that cannot create one
registers no CUDA devices, at which point a perfectly correct `--device CUDA0`
names a device that, for that process, does not exist.

None of that is a bug in the arithmetic that ran. It is all the same bug in
*when* it ran and *what it was allowed to see*.


## 2. The plan

`mc_plan.build()` assembles a `Plan` from the features that are actually on:

```
Creative Writer -> Spatial Composer -> Stage 1 (krea2) -> Handoff -> Stage 2 (klein9b) -> Stage 1 warm-up
```

Each `Phase` carries a kind and a peak. Preparation phases (`Creative Writer`,
`Spatial Composer`, `Direct BBOX Merge`) hold no image residency of their own —
their memory is llama-server's, which is the thing being budgeted *around*, and
counting it on both sides would reserve it twice.

Two rules produce the reserve:

**Max, not sum.** `Plan.image_working_peak()` takes the largest phase. Stage 1
and Stage 2 take over one another's arena; summing their residencies describes a
machine that runs both at once, which no code path here does.

**Real overlaps are counted.** The handoff keeps Stage 1's VAE and text encoder
resident while Stage 2 loads — that is `mc_memory._pinned_keep`, and it is
deliberate. So the handoff is a phase with its own peak, `Stage 2 + Stage 1's
module files`, and on a long chain it is frequently the largest one in the plan.

Stage 2 is sized at the resolution Stage 2 samples at, not Stage 1's. A 1.5×
multiplier is not a 50% error to ignore: activations scale with pixels, so it is
a 125% one.


## 3. Where the plan comes from

Both txt2img scripts publish it, and both build it from `p` alone:

- `mc_plan.stage_2_from(p)` reads the Model Chain panel's controls off
  `p.script_args`, located by script title;
- `mc_plan.creative_from(p)` does the same for the Creative panel — its enable
  flag is first and its three Spatial controls are last, which is what makes the
  variable-length axis block in the middle irrelevant.

Neither script chooses which hook the host runs first, so neither of them may
get a different answer. `tests/test_plan.py::TestReadingTheOtherScript` asserts
they do not, and `TestTheArgumentIndicesAreRight` asserts the positional indices
against the live panel rather than a copy of it: reorder the Model Chain UI and
a test fails, rather than a reserve being silently computed from a seed offset.

The Creative script passes `spatial_compose` explicitly because it alone knows
whether the layout it just parsed has boxes in it. Spatial Layout switched on
over an empty canvas runs no Composer, and a plan that said otherwise would name
a phase the user never sees.


## 4. The budget

```
ImageProtectedBudget = Plan.image_working_peak() + user safety adjustment
PersistentLLMBudget  = usable VRAM - ImageProtectedBudget   (capped by Custom)
```

The global safety margin is *not* added again: `mc_memory.vram_required_bytes`
has already folded it into every phase peak, and adding it twice sets a
gigabyte and a half of card aside twice for one activation peak.

`mc_llm_runtime._spendable()` is the single funnel every fit decision goes
through — `_fits`, `_shrink_context`, `_shrink_offload`, and the automatic
context sizing in `_requested_placement`. It is free VRAM, capped by the plan's
budget and by any ceiling a previous reserve miss taught. With no plan published
it is exactly `_free_vram()`, which is what every path did before.


## 4a. Obtainable VRAM, not the card's nameplate

`usable_vram_bytes()` measures rather than declares, and the first release of
this change did not — which was enough on its own to keep restarting the server
even with every other rule working.

A card's nameplate is not available to anything. The same user's log, at every
single start:

```
- CUDA0 : NVIDIA GeForce RTX 3090 (24575 MiB, 23304 MiB free)
```

24575 total, 23304 free with nothing whatever loaded. The missing **1271 MiB
(1.24 GB)** is the display, the driver's working set and the desktop, and it is
never obtainable.

Sized from the nameplate, `PersistentLLMBudget = total - protected` is 1.24 GB
too generous, so the model is *placed* 1.24 GB larger than the card can carry
beside the image plan. Nothing fails at that point: llama.cpp starts happily,
because the VRAM really is free while the checkpoint is not loaded. What fails
is the next question anybody asks — "does the image plan fit right now?" The
answer is no, by almost exactly the overshoot, and the only thing the broker can
do about it is stop llama-server. Every generation, with an identical placement
every time, because the overshoot is identical every time.

So the figure is measured from three terms that are all really ours:

```
obtainable = free to the driver, right now
           + what our own language model is holding
           + what our own image models are holding
```

The sum is invariant as models come and go — a checkpoint loading moves bytes
from the first term into the third and leaves the total alone — which is the
same stability the plan itself provides. It also means a *third-party* process
taking VRAM correctly shrinks the budget, which the nameplate could never see.

The invariant this restores: with the language model holding its whole
allowance, what is left on the card is still the protected peak, so nothing has
to be handed back and no eviction is reached.

`mc_plan._log_derivation` writes the whole sum to the console once per plan, so
a budget that looks wrong is wrong visibly:

```
Model Chain: memory budget — 22.8 GB obtainable of 24.0 GB on the card,
             17.0 GB protected for the image plan, 5.8 GB for the LLM (auto)
```

A large gap between the two first figures on any machine is the single likeliest
explanation for a language model that will not stay up.


## 4b. Measured peaks, not file sizes

`vram_required_bytes` is `file size x 1.15 + headroom`. It is a starting
heuristic and was always documented as one, and on a quantised,
mixed-precision checkpoint it is badly out. From the same user's console:

```
Model Chain: active plan — Creative Writer -> Stage 1 (kromaInt8ConvrotFor_v02Turbo);
             image working peak 21.4 GB, set by Stage 1
Model Chain: memory budget — 22.7 GB obtainable of 24.0 GB on the card,
             21.4 GB protected for the image plan, 1.3 GB for the LLM (auto)
```

Forge's own load log for that same generation:

```
Requested to load JointTextEncoder ... 5744.04 MB loaded
Requested to load KModel           ... 12234.77 MB loaded
Requested to load WanVAE           ...  242.03 MB loaded
```

17.8 GB actually loaded against 21.4 GB reserved — **3.6 GB of phantom
reserve**, on a card with 22.7 GB obtainable. The language model gets what the
image plan does not need, so those 3.6 GB were the whole difference between a
placement with layers on the GPU and this:

```
Model Chain: starting llama-server — Q3_K_P, no layers on the GPU
             (weights in system RAM), 8,192 token context
```

which reads prompts at 67 tokens a second instead of several hundred. The
Spatial Composer's 611-token prompt then takes 9.1 s to evaluate, which is the
"few seconds before the spatial prompt starts" — not a reload, not a cache
reset, just arithmetic on the wrong processor.

So a phase is measured wherever it can be:

1. **the loaded model, asked directly** — `mc_memory.measured_weight_bytes`
   sums the patchers' own `model_size()`, which is true right now;
2. **a remembered measurement** — every checkpoint records what it weighed the
   last time it was switched to (`mc_plan.remember_weights`, hooked into
   `_pass_requirement`, the one place a real figure for a named model exists).
   This is how Stage 2 gets a real peak in a plan built while Stage 1 is loaded;
3. **the file estimate**, only when neither of the above has ever seen it.

The measurement is keyed by checkpoint *and* its module files: a Krea or Flux
checkpoint keeps its VAE and text encoder separately, and on this setup those
two were 6 GB of the 18 GB total, so changing the text encoder changes what the
pass weighs. It is held in memory only — re-learned by the first generation of
any session, and a figure written to disk would outlive the file it describes.

`Phase.measured` carries which of the three answered, and the panel says so on
every image row. A table that presented an estimate with the same authority as
a reading would be inviting the user to trust the wrong one.


## 4c. A batch multiplies the activations

Every image in a batch carries its own latents, its own attention buffers and
its own residual stream, and they are all live at the same moment. The weights
are shared; nothing else is. So the plan takes `p.batch_size` and multiplies the
pixel count by it.

Deliberately **not** `n_iter`: four batches of two run one after another and
cost what one batch of two costs. Reserving for eight images would leave the
language model nothing, for nothing.

The bug this fixes killed a generation before its first step. A batch of five on
a 24 GB card holding a 17.8 GB checkpoint:

```
Reclaimed: 21640 MB    Residual: 1312 MB
Driver-free: 255 MB    Torch reserved: 21056 MB
Estimated next operation: 1352 MB
Trigger: driver-free memory below the hard floor (checked at unet-forward)
```

The plan had protected **19.3 GB** — weights plus *one* image's activations. The
pass reserved **21.06 GB**. The 1.76 GB difference is four extra images' worth of
activations, and a llama-server was sitting in 1.4 GB of it under a promise this
plan had made it. With Creative Mode off the same batch completed, because
nothing was holding that 1.4 GB.

The observation side is corrected too. `observe_activation_peak` now divides by
the batch, so what it learns is the cost of *one* image at that resolution.
Without that, one batch of five would teach the estimator a per-megapixel figure
five times too large and every single-image generation afterwards would reserve
for a batch nobody asked for.

### The allowance can now fall, and the server must follow it

`_worth_restarting` only ever asks for an improvement, and its reasoning was
sound: a running server holds its VRAM either way, and moving it somewhere
smaller frees nothing anybody asked for. A plan is exactly that request. When a
boundary lowers the allowance below what the server is holding,
`Runtime._overspending` returns true and the placement is redone downwards —
with a 256 MB tolerance, because both sides of that comparison are measurements
and re-placing for a rounding error is the flapping this work removed.


## 4d. Noticing an overrun at all

When an estimate is short, nothing in this extension is in the path. The host's
allocator spills, or a VRAM guard in another extension stops the generation at
`unet-forward` and reports it in its own words. Neither reports back, so the
panel went on quoting a budget that had just been exceeded.

`mc_plan.check_observed` closes that. `observe_activation_peak` already measures
the pass peak; it is now compared with what the plan protected, and a pass that
peaked above it is recorded as a reserve miss — `evicted=False`, because nothing
was taken. This is the miss being *noticed*, which is a different event from the
recovery.


## 4e. Where the card's memory actually is

The panel now shows a residency map that adds up to the whole card:

| | |
| --- | --- |
| Image models | weights as loaded, not as stored on disk |
| Language model | 0 when it is running from system RAM |
| Free | |
| Everything else | CUDA context, desktop, other programs |
| **Card total** | |

The last row is the one worth having. A user reported 20.1 GB in Task Manager on
an idle machine whose three model files come to 17.3 GB. All of it was ordinary:

| | |
| --- | --- |
| text encoder (4.88 GiB file) | 5.61 GiB loaded — fp8 file into bf16 compute |
| transformer (11.95 GiB file) | 11.95 GiB loaded |
| VAE (0.47 GiB file) | 0.24 GiB loaded — decoder only |
| **resident weights** | **17.79 GiB** |
| CUDA context | ~1.33 GiB, never returned until the process exits |
| llama-server, 6 layers | ~1.4 GiB |

Loaded size is not file size in either direction, and a CUDA context is over a
gigabyte before a single weight arrives. None of that was visible anywhere,
which is the part that was actually wrong.


## 4f. A restart must be worth the cache it destroys

`CONTEXT_UPGRADE_FRACTION` had this rule from the start — a quarter more context
before a running server is replaced, because a restart re-reads the weights and
empties llama.cpp's prompt cache. The **layer** comparison beside it had no such
rule, so any gain at all was acted on.

From a user's console, with timestamps:

```
[00:22:48.743] re-placing llama-server — 2 of 30 layers on the GPU, experts in
               system RAM now fits, where it is running no layers on the GPU
[00:22:51.115] llama-server stopped — making way for a new placement
[00:22:58.661] llama-server ready — 2 layers on the GPU ... 0.9 GB VRAM
```

**9.9 seconds** to gain two blocks of thirty. llama.cpp's own timings either
side:

| | prompt | generation |
| --- | --- | --- |
| 0 layers | 61.48 tok/s | 12.71 tok/s |
| 2 layers | 62.05 tok/s | 12.46 tok/s |

Inside the noise, and slightly worse on the half that matters. And because every
start begins `cache state: 0 prompts`, the writer's 523-token prompt was read
from scratch for the second time in a minute — a further 8.4 seconds. Eighteen
of the fifty-three seconds that "warm" generation spent on the language model
went on undoing its own warmth.

`_worthwhile_layer_gain` now requires a quarter of the model's blocks, never
fewer than four. The downward direction is untouched: `_overspending` is about
correctness, not speed, and still acts on any real overshoot.


## 4g. The measurement outlives the session

`mc_plan` keeps measured checkpoint weights in
`model_chain_weights.json`, beside the presets and the timing calibration in the
WebUI data directory.

It used to hold them in memory only, on the reasoning that a figure written down
outlives the file it describes. The cost of that showed up in the same console:

```
[00:21:19.605] image working peak 21.4 GB (estimated from file sizes)
[00:22:47.477] image working peak 19.3 GB (measured)
```

Generation one planned from file sizes, generation two from the measurement, and
the 2.1 GB between them moved the plan boundary — which is what re-placed
llama-server above.

The key answers the original objection: it carries the **total size of the
checkpoint and its modules**, so re-quantising a checkpoint under the same name
changes the key and the stale figure is never found again.


## 4h. Believing a residency of zero

The ready line said `0.0 GB VRAM` about a placement llama.cpp was running at 108
tokens a second on prompts — a model holding roughly fourteen gigabytes. The
residency is a difference of two free-VRAM readings, which is the only
measurement available from outside another process, and the second reading was
taken before the driver had caught up: the health endpoint answers as soon as
the server will accept a request, and nothing obliges the free figure to be
current by then.

That zero is not cosmetic. It is the figure every later question about whether
that server is overspending its allowance starts from.

`Runtime._observed_residency` now retries a non-positive reading, four times at
a quarter-second, and only on a placement that is supposed to be holding
something. A placement with no layers on the card reports zero at once, because
there zero is the answer.


## 5. Stability

`Runtime._outgrown` is where re-placement was decided, and it now declines to
ask the question at all in two cases:

- an image job holds *this server's* card (`Runtime._image_job_conflicts()`),
  because halfway through a generation there is always memory that has just
  been released and is about to be taken again, and every one of those instants
  looks like room to grow into. Scoped to the card since the resource-scoped
  concurrency work — see `docs/12-resource-scoped-concurrency.md` — because a
  generation on GPU 0 is no reason to freeze a placement decision on GPU 1;
- the plan the running server was placed for is still the plan in force
  (`Runtime._boundary_moved()` is false). That baseline lives on the runtime
  rather than in `mc_plan`, so two servers cannot answer each other's question.

`Plan.identity()` quantises phase peaks to quarter-gigabyte classes. A
byte-exact comparison would call every generation a new plan, because an
observed activation peak moves a little on every pass — and a plan that is new
every generation restarts llama-server every generation, which is the behaviour
being removed.


## 6. One server for consecutive LLM calls

`Creative.roll` used to hand the card back to the image side before yielding its
prompt. With Smart Spatial on, the Spatial Composer runs immediately afterwards,
and the only way the broker can free image VRAM is to stop llama-server — the
very process the Composer was about to send its one request to.

So the hand-back is deferred: `mc_creative_krea._composer_follows(layout)` is
true only for a Smart layout that actually has regions, and the script performs
the hand-back after the Composer instead, whether or not the Composer succeeded.
The rule generalises to any persistent service: do not reclaim it between two
consecutive operations that both need it.


## 7. When the estimate is wrong

The image generation wins. `mc_memory.make_vram_room` frees optional image-side
state first; if the pass still does not fit it reaches
`mc_memory._reclaim_foreign`, which is where llama-server is evicted.

Reaching there at all is a reserve miss, so it is filed as one *before* anything
is taken — the LLM's residency is read first, because after the eviction the
answer is zero for every miss and the panel could not tell a model that gave up
eight gigabytes from one that was never on the card. `mc_plan.record_miss`
records the phase, the shortfall, whether an eviction happened, and a suggested
safer cap half a gigabyte below what would have fitted. That figure becomes a
ceiling on the next automatic allowance, so the model is not silently promoted
back to a placement that failed.


## 8. Requirement mapping

| § | Requirement | Where |
| --- | --- | --- |
| 3 | Build the active execution plan from the features that are on | `mc_plan.build`, `mc_plan.build_for` |
| 5–6 | Phase-based accounting; peak of plan, not sum | `Phase`, `Plan.image_working_peak` |
| 7 | Two complete image models are not assumed to coexist | `TestScenarioEBothStagesNoCreative` |
| 4D, 7 | Real transition overlaps counted | the `HANDOFF` phase, `_module_bytes` |
| 8 | `PersistentLLMBudget` is the remainder | `mc_plan.persistent_llm_budget` |
| 9–10 | The LLM floor holds; no growth into transient free VRAM | `mc_llm_runtime._spendable` |
| 11 | Re-optimise only at plan boundaries | `mc_plan.boundary_moved`, `Runtime._outgrown` |
| 12–13 | One warm server for every LLM call in the plan | `mc_creative_krea._composer_follows` |
| 15–16 | Warm-up is a phase and yields rather than flapping the floor | the `WARM_UP` phase |
| 17 | Emergency eviction as recovery, recorded | `mc_memory._record_reserve_miss` |
| 18 | Start failures diagnosed rather than collapsed | `mc_llm_runtime._DEVICE_GONE` |
| 19–25 | One txt2img section; plan, peaks, budgets, LLM values, cap, misses | `mc_plan_panel.py` |
| 26–27 | The generic rule set and scenarios A–H | `tests/test_plan.py` |
| 28 | Logging on every plan and placement decision | `mc_plan.publish`, `Runtime._record` |
