# Item 4 — Hardware-aware, self-calibrating whole-job progress and ETA

Project: SD-Neo-ModelSwitchRefiner ("Model Chain")
Host: Stable Diffusion WebUI Forge Neo (`neo`)

**Revision 2.** The first draft was written against `claude/new-session-24dap2`,
before the VRAM work landed (`29a3b5f`, `3ca0a1b`, `73e3f7e`, `60ec527`,
`0c3f455`). That work changed the phase structure this item models, so three of
the original requirements were wrong rather than merely incomplete. Those
corrections are marked **[was]** below.

Implemented in `mc_progress.py`, wired from `scripts/model_chain.py`, covered by
`tests/test_progress.py`.

---

## Background

Forge Neo computes progress in `modules/progress.py`:

```python
if job_count > 0:
    progress += job_no / job_count
if sampling_steps > 0 and job_count > 0:
    progress += 1 / job_count * sampling_step / sampling_steps

predicted_duration = elapsed_since_start / progress
eta = predicted_duration - elapsed_since_start
```

That describes a single sampling loop well and a chained job badly. Two things
follow from the formula.

**It reaches 100% and falls back.** Stage 2 used to be added to `state.job_count`
in `postprocess`, by which point Stage 1 had already driven `job_no` up to the
old count. The spec's own worked example — batch of two, Stage 1 20 steps,
Stage 2 8 steps — went **100% → 33% → 66% → 100%**.

**It weights unlike work equally.** One 8-step Stage 2 refine counted for the
same third of the job as the entire 20-step, two-image Stage 1 batch. Time spent
moving several gigabytes of weights between them counted for nothing at all.

## The one thing that has to be right

`eta` is derived from `progress`. If `progress` is *the fraction of predicted
wall time spent* rather than the fraction of steps taken, the host's ETA becomes
correct as a consequence, and there is no second, separately-wrong calculation
to maintain.

**[was] "Progress must be monotonic and should correspond to predicted remaining
wall time."** Split into two requirements that were being conflated. Monotonicity
is a guard. Time-proportionality is the model. The implementation supplies both
`progress` and `eta` from the same arithmetic rather than letting the host
re-derive one from the other, because `state.time_start` is stamped slightly
before the extension's first phase and the offset is visible on short jobs.

---

## Phases

A job is an ordered list of phases, each with a predicted duration, entered by
key as the job reaches it. Entering a phase closes every phase before it, so a
phase that turns out not to be needed costs nothing and shortens the prediction.

| Phase | What it covers | New since revision 1 |
| --- | --- | --- |
| `join` | `mc_memory.join_preload()` — waiting for the previous generation's background preload | **yes** |
| `stage1_prepare` | `reinstate_pending`/`consume_preload` and `_make_room_for_stage_1` | **yes** |
| `stage1` | every Stage 1 sampler pass, including the hires second pass, and the decode after each | |
| `switch` | `mc_memory.ensure_resident()` — the one checkpoint switch | |
| `stage2_prepare` | `mc_memory.make_vram_room()` before the refine | **yes** |
| `stage2` | one sampler pass per image | |
| `finalize` | reference cleanup, selection restore, assembly and saving | |

### What the VRAM work changed

**[was] "optional Stage 1 restore/preload" as a job phase.** It cannot be one.
`_preload_stage_1()` runs on a background thread from the `finally` block of
`_run_stage_2`, and `postprocess` returns immediately afterwards. The host marks
the task complete when the Gradio call returns and `javascript/progressbar.js`
removes `.progressDiv` from the DOM the moment `res.completed` is true. There is
no bar left to report a preload on.

**Resolved:** *job complete* means the final images are returned. The preload's
state is reported by the panel's readiness line, which `mc_memory.stage_1_readiness()`
already provides. The original acceptance bullet asking for this decision to be
made is answered here rather than left open.

**[was] model movement as a single learned cost.** It is one of several
residency kinds, and `mc_memory.plan()` predicts which one *before the job
starts*. They differ by an order of magnitude — the module's own docstring cites
0.8s against 11.5s for the same 8 GB UNet under different headroom. The estimate
keys on `plan().kind`, taken from the same call that produces the console
explanation, so the prediction and the explanation cannot disagree.

Note that `dual` says nothing about where the model is coming *from* — it means
it fits alongside Model A without an eviction, and is still a first load. Its
baseline is a disk-read rate, not a free one.

**[was] "bytes to move" as file size.** `capture_stage_1_encoders` pins Stage 1's
VAE and text encoder through the switch, so a warm swap moves only the UNet.
Sizing from `file_size_bytes` alone over-estimates every warm swap.

**Also new:** `join_preload()` blocks at the very front of `before_process`,
before the extension knows whether the chain is armed. That time is real, it is
spent staring at an empty bar, and it is now timed and handed to the estimator
retroactively (`Job.record`, `begin(since=...)`).

### Hires fix

Not mentioned in revision 1. It matters twice:

- `StableDiffusionProcessingTxt2Img.init` doubles `state.job_count` and calls
  `total_tqdm.updateTotal` for a hires job. The extension claims that refinement
  (`state.processing_has_refined_job_count = True`) and supplies both numbers
  itself, rather than compensating for the doubling afterwards.
- The two passes cost different amounts and *interleave* through a multi-batch
  job (first, hires, first, hires). Each pass carries its own weight, so the bar
  does not speed up and slow down for a reason the user cannot see.

---

## Estimation

### Units

Rates are stored per unit of what the phase actually does, so a rate learned at
one size predicts another:

| Rate key | Unit |
| --- | --- |
| `move:<kind>` | seconds per gigabyte |
| `free` | seconds per gigabyte |
| `sample:<arch>:b<batch>` | seconds per step per megapixel per image |
| `join`, `finalize` | flat seconds |

### Batch count and size

Stage 1 samples `batch_size` images in one pass, `n_iter` times. **Stage 2
refines each image as its own `process_images` call with `batch_size=1,
n_iter=1`** — so a chain gets none of Stage 1's batching gain, and predicting it
as though it did under-estimates badly at batch 4 and above.

Batching is sublinear on real hardware, so the sampling rate key includes the
batch size. A rate measured at batch 4 is kept apart from one measured at batch
1 and is only fallen back to when there is nothing better.

### First run

A small built-in table, tilted by detectable hardware: the sampling baseline
scales by total VRAM against a 24 GB reference and is clamped hard at both ends.
VRAM is a poor proxy for compute, but it is the one figure reliably readable
offline on every platform the host runs on, and being roughly right on an 8 GB
card beats being confidently wrong.

No DDR generation, no bus width, no benchmark database, no network. An unseen
transition kind is costed as the slowest one — an under-estimate reads as a
hang, an over-estimate reads as slack.

### Self-calibration

Every phase is timed as it runs and folded into the store with a smoothing
weight of 0.4, so recent comparable jobs outweigh generic guesses without one
contended pass becoming the new truth. The first observation for a key is taken
whole rather than averaged with a guess that was never about this machine.

A phase with no units to measure, or a duration of zero, teaches nothing — a
skipped VRAM free is not evidence that freeing is free.

**Interrupted and failed jobs are dropped, not folded in.** A phase cut short
measures the interruption, not the work.

**[new] Measurements persist.** `model_chain_timing.json` in the WebUI data
directory, beside the presets, written through a temporary file and an atomic
replace. `observe_activation_peak`'s learned VRAM reserve is in-memory only and
resets every restart; if timing did the same, "after several similar jobs the
ETA becomes measurably closer" would only hold within a session.

### Within a phase

- **Sampling** interpolates from `state.sampling_step / state.sampling_steps`
  and the completed passes' weights, then extrapolates the remainder from what
  the phase has *actually* cost so far. A bad initial estimate self-corrects
  mid-phase.
- **A sampling phase reserves a 12% tail.** A pass is not over when its last
  step is — the latents still have to be decoded, and the host exposes no
  counter for that. Without the tail the bar stalls on a full step count at
  every pass boundary.
- **Opaque phases** — weights moving, VRAM being freed — have only the estimate.
  A shrinking remainder is held back so an overrun keeps counting down rather
  than stalling at zero, which is indistinguishable from a hang.

---

## Delivery to the host

### Primary: wrap `progressapi`

`webui.py:137` calls `progress.setup_progress_api(app)` long after
`initialize.initialize()` has loaded extension scripts, and `add_api_route`
resolves `progressapi` from the module globals at that moment. Rebinding the
global at extension import time is therefore picked up.

The original is called first and only `progress`, `eta` and `textinfo` are
overwritten. Live previews, queueing and task bookkeeping stay the host's.

Two details that are easy to get wrong:

- FastAPI builds the request model from the function's annotation. `mc_progress`
  uses postponed annotations, so a written-out annotation would arrive as the
  string `"ProgressRequest"` and be resolved against the wrong globals. The
  class object is assigned to `__annotations__` directly.
- `ProgressResponse` is the declared `response_model`, so extra fields are
  filtered out. Anything custom has to travel in `textinfo` — and the host's JS
  drops a `textinfo` containing a newline, so phase labels are kept to one line.

A plan left behind by a generation that raised between `process` and
`postprocess` is recognised by comparing its start against `state.time_start`
and dropped rather than used to describe the next task.

### Fallback: pre-size the host's counters

Applied whether or not the wrapper installed, because it also fixes the console
bar. `state.job_count` and the `total_tqdm` total are set to their final values
in `process()`, before any sampling. This alone removes the backwards jump; it
cannot weight phases by time or represent model movement.

### Phase labels

`shared.state.textinfo` reaches the bar unmodified and nothing else sets it
during generation. `"Stage 2 1/2"` counts through the batch without the
percentage reacting to it.

---

## Acceptance

- [x] Batch count 1, batch size 2, Stage 1 = 20 steps, Stage 2 = 8 steps
      produces one continuous 0–100% trajectory across both images and both
      stages.
- [x] The bar does not hit 100% until the job is complete — it never reports
      completion at all; the host removes the bar when the task ends.
- [x] Model movement contributes to ETA and percentage, costed by the residency
      kind that was predicted for it.
- [x] After several similar jobs the ETA is measurably closer, and stays closer
      across a restart.
- [x] First-run estimation works with no network.
- [x] If hardware detection fails, progress still works from the flat baselines.
- [x] A useful phase label is exposed without the percentage jumping.
- [x] Hires fix, batch count and batch size are all modelled.
- [x] "Job complete" means the images are returned; the preload is reported by
      the panel, not the bar.

## Watchpoints for future work

- The `progressapi` rebind is the fragile part. It is wrapped, guarded, and
  falls back to the counter fix — but a host change to how the route is
  registered would silently revert the bar to the host's own numbers, with no
  error.
- The estimator must never affect what comes out of the pipeline. Every entry
  point swallows its own exceptions; a planning failure leaves the generation
  untouched.
- Different backends change the phase structure. Phases are declared by key in
  one ordered list (`mc_progress.ORDER`); adding one is a list entry and an
  `enter()` call.

## Relevant code

- Model Chain: `mc_progress.py`, `scripts/model_chain.py`, `mc_memory.py`
- Forge Neo: `modules/progress.py`, `modules/processing.py`,
  `javascript/progressbar.js`, `backend/memory_management.py`
