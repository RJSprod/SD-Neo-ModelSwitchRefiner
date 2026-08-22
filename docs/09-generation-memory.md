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


## 5. Stability

`Runtime._outgrown` is where re-placement was decided, and it now declines to
ask the question at all in two cases:

- an image job holds the card (`mc_broker.host_busy()`), because halfway through
  a generation there is always memory that has just been released and is about
  to be taken again, and every one of those instants looks like room to grow
  into;
- the plan the running server was placed for is still the plan in force
  (`mc_plan.boundary_moved()` is false).

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
