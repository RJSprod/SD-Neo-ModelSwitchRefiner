# Voice Box — VibeVoice 7B in Forge, and whose turn it is on each card

The revised design for the Voice Box tab. The product intent is the user's
design-intent document of 2026-09-28, kept beside this file as
`docs/23-voice-box-intent-2026-09-28.txt`; section numbers written *§n* below
are that document's. Everything here was settled with the user on 2026-09-29,
after the feasibility review (the evidence is in handoff 31), and where the two
differ this file wins.

Behaviour people see is in `README.md`. The reasoning behind a particular line
of code is in that module's docstring, which is where it stays true.

---

## 1. What Voice Box is

A native Forge tab that makes Microsoft's VibeVoice a speech workspace: long,
expressive, multi-speaker speech with reference-audio voices, on a graphics card
the user picks, coordinated with everything else that wants that card.

The model is **VibeVoice 7B** ("Large"). The 1.5B is the same runtime with a
smaller checkpoint and stays a supported choice; the Realtime 0.5B is a later,
separate mode (§3C, §15), because its voices are preset-only by design.

The page is a pipeline without wires, in four stages:

    INPUT  ->  PROMPT  ->  CONFIGURATION  ->  OUTPUTS

and several pipelines can be open at once, the way conversations are.

---

## 2. Decisions recorded with the user

| Question | Decision |
|---|---|
| Model | VibeVoice 7B. One-click download from a community mirror, every file checked against a SHA-256 committed here, plus install-from-folder as a fallback |
| Which card | Chosen per pipeline: the 3090 or the 5090. The integrated Arc is out of scope |
| When a request arrives | It becomes the **next** job on its card. The job running there is never cut off |
| Several VibeVoice requests | All of them go before any waiting image, WanGP or LLM job |
| Long renders | Run to the end in one generation; waiting jobs wait for the whole render |
| Renders at once | One per card; the two cards may each run one |
| Between requests | VibeVoice stays warm in VRAM while nothing else waits for its card, and is evicted (only while idle) when an image job, WanGP or the LLM needs the card |
| The image model it displaced | Comes back. Setting: **Automatic** (default), From system RAM, From recipe. Never a temporary file on disk |
| Running out of VRAM or RAM | **Never the pagefile.** A VibeVoice request that would overflow either is blocked with a warning naming the shortfall. Scoped to VibeVoice for now; everything else keeps today's behaviour |
| WanGP's card | Mini Paint holds its next Clipboard job; the WanGP bridge holds WanGP's own queue; WanGP is flushed only when the 7B does not fit beside it |
| img2img and inpaint | Held exactly as txt2img is |
| Inputs | Played and trimmed locally in the browser; only the trimmed sample is sent. Upload is the fallback for files the browser cannot decode |
| Audio focus | One speech source at a time, two-way, with Voice Chat only |
| Saving | A folder on the Forge PC chosen once and remembered, plus Download |

---

## 3. The model and its runtime

**Weights.** Microsoft deleted VibeVoice-Large from Hugging Face on 2025-09-04;
it survives as MIT-licensed community copies (`vibevoice/VibeVoice-7B`,
`aoi-ot/VibeVoice-Large`, 18.7 GB in bf16, 9.34 B parameters). The 1.5B is still
Microsoft's (`microsoft/VibeVoice-1.5B`, 5.4 GB). Neither repository contains a
tokenizer: the processor fetches `Qwen/Qwen2.5-7B` (or `-1.5B`) from the hub
unless handed a local directory, so the installer ships the four tokenizer files
(`tokenizer.json`, `tokenizer_config.json`, `vocab.json`, `merges.txt`) and the
worker is started offline (`HF_HUB_OFFLINE`, `TRANSFORMERS_OFFLINE`), as every
voice worker already is.

Using a third-party mirror is a trust decision the user made explicitly. It is
made safe the way every other artifact here is: pinned by SHA-256. The machine
that writes this repository cannot reach Hugging Face, so the pins are produced
by a `tools/pin_vibevoice_models.py` run on the user's machine and committed; an
install before that is checked against the publisher's own LFS digest and
recorded in the untracked overlay, exactly as `README.md` "Who vouches for the
bytes" describes.

**Code.** Microsoft removed the long-form TTS code. The runtime is the MIT
community package (`vibevoice` on PyPI, the preserved original), because it
reads Microsoft's checkpoints directly, streams audio every 133 ms frame through
an `AudioStreamer`, and checks a `stop_check_fn` at the top of every generation
step. transformers 5.17 has a native port, but it hands its streamer token ids
and the audio only at the end, and needs converted checkpoints.

The community code breaks on transformers 4.56 and later (`DynamicCache.key_cache`
was removed), so its closure pins `transformers==4.51.3` and `accelerate==1.6.0`,
and therefore `diffusers` below the release that needs `huggingface-hub>=1.23`.
The demo-only dependencies (`gradio`, `aiortc`, `av`, `ml-collections`,
`absl-py`) are not installed.

**Torch.** One CUDA 12.8 build serves both cards: the Windows cu128 wheels carry
sm_86 (the 3090) and sm_120 (the 5090) from torch 2.7. torch and torchaudio are
pinned as a *pair* by version and digest. The existing cu128 entry for LavaSR
resolves both unversioned and separately, which can mismatch; the VibeVoice
closure does not copy that. `PYTORCH_NO_CUDA_MEMORY_CACHING`, which the CPU
engines set, is not set here: on a card it would disable the caching allocator.

**Attention.** SDPA. Microsoft's demo calls flash-attention the only fully tested
path, but flash-attention has no official Windows wheels and nobody has measured
a difference; the first renders are compared by ear.

**Length.** The 7B's context is 32K, about 45 minutes. `generate()` also stops at
`max_length_times` (2) times the prompt length; the runtime passes a value that
lets a script reach its end.

**Provenance.** The model card promises an audible disclaimer and a watermark;
neither is in the open code (they were the hosted demo's). Voice Box writes its
own: every output carries its model, voices, seed and settings (§16), saved
voices are labelled as cloned from a recording (§17), and nothing is ever made
from a recording the user did not choose.

---

## 4. Whose turn it is on each card

### 4.1 The rules

Stated for one physical card, identified by UUID.

1. **A running job is never cut off.** An image job (one Generate press, whole
   batch), a WanGP task, an LLM turn and a VibeVoice render all run to their end.
2. **A VibeVoice request goes next**, ahead of every job on its card that has not
   started.
3. **All VibeVoice requests first.** Consecutive requests on one card run back to
   back before any waiting job.
4. **A render runs to its end** in one generation.
5. **One render per card.** The two cards may each run one.
6. **VibeVoice stays warm** on its card while nothing else waits for that card,
   and is evicted, only while idle, when an image job, WanGP or the LLM needs the
   card, or when system RAM gets tight while the image model is parked for it.
7. **What made room comes back** (section 4.3).
8. **Never the pagefile** (section 5).

### 4.2 The lifecycle of a VibeVoice turn

    queued -> clearing -> making room -> granted -> (next request | warm | done)
                                   \-> blocked (with a warning)

*queued*: waiting behind another VibeVoice request on the same card.
*clearing*: the card's gates are closed to work that has not started, and the
turn waits for the work that has. *making room*: the image model is parked, an
idle llama-server on the card is stopped, WanGP is flushed if it must be, and the
card is measured. *granted*: VibeVoice runs. *warm*: VibeVoice is resident and
idle. *blocked*: admission refused; the warning says what was short and by how
much, and nothing that moved to make room stays moved.

### 4.3 The image model's card (the 3090)

**The gate.** Every txt2img and img2img generation — inpaint is img2img — calls
the gate first thing in `before_process`, before Model Chain's warm-up, preload
and budget. Model Chain's own script is txt2img-only, so a separate always-on
script carries the gate on both tabs, and Model Chain's hook calls it too, so the
order of the two scripts does not matter. The gate is idempotent per job.

While a VibeVoice turn is queued or running on the image card, the gate holds the
job: one line on its progress bar says it is waiting for VibeVoice, and the gate
never sets `state.job` or `job_count` — a job that sets them itself deadlocks the
host. Interrupt and Skip are honoured but do not cut the wait short: the job ends
as soon as the card is free, and the progress line says so. Let go early, it would
still load its checkpoint — and img2img encodes its input on the card in `p.init`,
before the host's first interrupt check — on top of a render rule 1 says may not
be cut off. The host swallows an exception raised from `before_process`, so
raising is not a way out either.

**The current job.** A VibeVoice turn waits until nothing is running on the host
(`shared.state` idle and no `progress.current_task`) or the job that is running
is parked at the gate. A batch is one job, so a batch of ten finishes first.

**Ungated host work.** Extras, other extensions' pipelines and the API's
postprocessing do not pass through `before_process`. Every one of them takes
Forge's `modules.call_queue.queue_lock`, a FIFO lock, so for the length of a
render the turn holds that lock too: taken *without waiting* when it is free,
never queued for (queueing would put VibeVoice behind jobs already waiting, which
is rule 2 broken). When a job is parked at the gate, it is already holding the
lock, and nothing else can start either. The lock is released when the turn ends
or goes warm, and is never held across a host hook.

**Parking the image model.** Setting *Bring the image model back after
VibeVoice*:

| Choice | During the turn | Afterwards | Cost |
|---|---|---|---|
| **Automatic** (default) | In system RAM when it fits above the margin; otherwise by recipe | Whichever was used | — |
| **From system RAM** | The same model object, weights moved to system RAM by Forge's own unload (`ModelPatcher.detach`, which moves and frees nothing); checkpoint, VAE, text encoders and whatever LoRA state it had | Moved back; the checkpoint is not reloaded from disk | ~15 s out, 17–20 s back |
| **From recipe** | Dropped with Forge's own `unload_model_weights`, which keeps the loading parameters | Reloaded from its own files | 20–30 s, LoRAs re-applied by the next prompt |

From system RAM never falls back silently: if RAM cannot take the model above the
margin, the VibeVoice request is blocked with a warning that names the setting.
Nothing is written to disk on either path.

**The return.** When the turn ends and VibeVoice does not stay warm on this card,
the image model is warmed back straight away through the existing warm-up
(`mc_arm`), which moves resident weights back or reloads from the recipe and
clears Forge's global-unload flag correctly (handoff 27). When an image job was
waiting at the gate, that job loads its model itself, as it always does, and no
warm-up is started beside it: two loads of one model at once is the thing to
avoid. A job that arrives while the warm-up is still loading waits for it at the
gate. When VibeVoice does stay warm, the image model comes back with the next
image job: the gate evicts VibeVoice first and the job loads its model as it
always does. When VibeVoice is evicted because an image pass or the LLM needed
its room, the image model is not warmed back then: the room is for them. The
record that it was parked outlives that eviction, so the next VibeVoice turn on
the card brings it back when it ends, and the next generation, which loads its
own model, clears the record at the gate. Whether a turn parks anything is
decided by what is resident on the card, never by the record.

**The end of a turn.** The turn stays the card's until its guest's memory is
gone: the gate, the language model's wait and Model Chain's own warming all read
"the card is free" from the same state, and an eviction takes as long as a
worker process takes to let go of eighteen gigabytes.

**RAM while parked.** A parked model holds its size in system RAM for as long as
VibeVoice stays warm. If available RAM falls below the margin in that time,
VibeVoice's warm stay ends and the image model goes back to the card, which frees
the RAM. (If VibeVoice is rendering, it is not cut off; admission already checked
the room before the render began.)

**Model Chain's own background work** — the post-generation preload and the
startup warm-up — does not move weights onto a card a VibeVoice turn holds.

### 4.4 WanGP's card (the 5090)

Mini Paint NEO's Clipboard hands WanGP **one job at a time** and keeps the rest in
its own outbox, so between two videos WanGP is actually idle. "VibeVoice goes
next" is therefore mostly a gate in Mini Paint's executor, and WanGP's internals
are needed only for work started inside WanGP itself.

The turn asks Mini Paint for the card through an in-process **lease** (section 6):

1. The lease is requested; Mini Paint closes its executor gate at once, so no
   further Clipboard job is submitted.
2. The job WanGP is running finishes (rule 1).
3. Mini Paint asks the bridge to **hold** WanGP: WanGP's own between-task pause
   flag, and, when no WanGP worker is running, an idle claim on WanGP's GPU lock,
   so work started in the WanGP tab waits too.
4. The lease reports **held**. The turn measures the card; if the render does not
   fit, it asks for a **soft flush** (WanGP's weights stay in RAM), waits until the
   lease says held again and measures again, then does the same with a **hard
   flush** (WanGP's weights are discarded and its next task reloads them from
   disk), and blocks with a warning — Mini Paint's own reason in it — if it still
   does not fit.
5. After the render the lease is kept, renewed, while VibeVoice stays warm, and
   released when it does not. The lease is renewed every two seconds during the
   render and every second during a warm stay — not on every tick, because a
   renewal is a call into Mini Paint that rewrites its record.
6. A warm lease reports **wanted** when a Clipboard job is waiting at the gate or
   work in WanGP is waiting on the claim. The turn then evicts VibeVoice, confirms
   the card has its memory back — WanGP sizes itself against what the card reports
   free and has no fallback — and releases the lease.

Mini Paint's executor holds a job for a lease with no 60-second limit. A lease
that is not renewed expires on its own and everything resumes, so a Forge that
died cannot leave WanGP held.

Without Mini Paint there is no WanGP to coordinate with. With a bridge older than
the hold capability, Clipboard jobs are still held but WanGP's own Generate is
not, so a render on WanGP's card is allowed only while WanGP is not running, and
a warm stay there ends the moment WanGP starts. A Mini Paint that reports WanGP
on the card but offers no lease — one older than the lease, or a record outside
the contract — is treated the same way, and no warm stay is kept on WanGP's card
without a held lease at all: the lease is the only thing that says when WanGP
wants its card back. A lease that stops answering is asked for again; Mini Paint
hands the same owner its live lease. A WanGP started by hand, outside Mini
Paint, cannot be coordinated at all; its memory still counts as not free.

This is a deliberate exception to handoff 28's rule that WanGP's VRAM is never
taken: a VibeVoice turn takes the card *between* WanGP's jobs, by WanGP's leave.
`mc_wangp`'s watch leaves a VibeVoice turn alone.

### 4.5 The language model

On the user's machine the LLM runs on the integrated Arc, which no VibeVoice turn
touches: an Intel GPU's model memory is host RAM, and it is in no card's VRAM
register. When a llama-server shares the turn's card: running LLM turns on that
card finish first; an idle server there is stopped; new LLM turns there wait for
the render; and an LLM request that needs the card while VibeVoice is warm there
evicts VibeVoice first.

One LLM turn does not wait for a turn that is still queued or clearing: the one
the host's own job is blocked on (Krea's writer, inside `before_process`),
because the clearing turn is waiting for that very job. Such a turn can
therefore begin between the clearing step and the next, so *making room* looks
for a running LLM turn again before it stops anything on the card, and waits
for it.

Both sides compare cards by UUID, and a UUID has two spellings: nvidia-smi's
`GPU-…`, which the LLM's setup records, and torch's bare digits, which the image
side records. `mc_broker.cuda_execution` reduces both to the digits; before it
did, one card in its two spellings was two cards, and an LLM on Forge's own card
was told it was independent of the generation there.

### 4.6 Where it lives in the code

`mc_turns.py` owns the turns. It is not a voice module: it imports the broker,
the memory side and `mc_wangp`, and the VibeVoice runtime never imports it. The
runtime is registered with it as a *guest* by `scripts/model_chain.py` and is
handed a client object — the direction of dependency that keeps invariant I-3
(`tests/test_voice_independence.py`) true for every CPU engine.

`mc_broker` gains a third family, `voice`: the image side's reclaim and the LLM's
look there first, a warm guest is evictable and a rendering one is not, and its
bytes are never reported as stray.

---

## 5. Never the pagefile

For every VibeVoice request, before anything moves:

* **RAM**, first, before anything is moved: VibeVoice's worker must fit within
  available RAM above a margin larger than the host's 2 GB floor, because Windows
  starts trimming working sets before memory is exhausted. Short: blocked, with
  the numbers, and the image model still on its card. Then — when the image model
  is to be parked in RAM — the model's resident bytes on top. Short: Automatic
  parks by recipe instead; From system RAM blocks.
* **VRAM**: after the card is cleared, the driver's free figure for that card
  must cover the render's estimate plus a margin. Short: blocked, with the numbers.
  A figure of zero is no reading, not a shortfall: the request goes ahead on the
  estimate, the log says so, and WanGP is not flushed for it.
* **During a warm stay**: section 4.3's RAM watch.

VibeVoice loads its weights straight onto the card; a full copy never passes
through system RAM.

Everything outside VibeVoice keeps today's behaviour: an image pass that does not
fit can still spill into shared memory, and the LLM's placement ladder still
ends in system RAM.

---

## 6. The contract with Mini Paint NEO

Two additions on Mini Paint's side, both version-gated and additive.

### 6.1 The lease: `minipaint_neo.wangp.turns`

Looked up by name in `sys.modules`, exactly as `presence` is. Every function
returns at once and never raises; none of them may be called from an ASGI handler.

```
request(owner: str, *, purpose: str = "", need_bytes: int = 0) -> dict
state(lease: str) -> dict            # also renews the lease
flush(lease: str, level: str) -> dict  # "soft" | "hard"; only while held
release(lease: str, *, reason: str = "") -> dict
report() -> dict                     # WanGP's activity, with or without a lease
```

A lease record has exactly these keys:

| Key | Meaning |
|---|---|
| `version` | 1 |
| `lease` | an opaque id |
| `owner` | who asked |
| `phase` | `pending`, `holding`, `held`, `released`, `expired` or `refused` |
| `reason` | one sentence when refused, expired, or still waiting |
| `card_uuid` | WanGP's card |
| `wanted` | true while a Clipboard job waits at the gate or WanGP work waits on the claim |
| `bridge_hold` | `none`, `holding`, `held` or `unsupported` |
| `expires_in_s` | seconds before an unrenewed lease expires |
| `wangp` | `{running, task_running, queue_length, jobs_waiting, vram_free_bytes, vram_total_bytes}`; any of them `None` when unknown |

*pending*: a WanGP task of Mini Paint's is still running. *holding*: the bridge
has been asked to hold WanGP and has not confirmed it. *held*: nothing is running
on WanGP's card for WanGP and nothing will start; the owner may use the card.
*released*/*expired*: the executor gate is open and the bridge has been told to
resume. *refused*: the integration is off, or WanGP is not Mini Paint's to hold;
`reason` says which.

One lease at a time. A second request by the same owner returns the live lease.
A lease not renewed for 20 seconds expires.

Three properties of Mini Paint's implementation the turn system is written
against (its `docs/wangp/CONTRACTS.md` states them as rules):

* **`flush` is asynchronous.** It returns at once with the lease `holding`;
  Mini Paint's executor does the flush on its next pass, and `held` again means it
  is done, or refused, with `reason` saying which. The turn measures the card
  after `held`, never after `flush` returns — measured earlier, the card reads a
  WanGP that has not started letting go.
* **Phases are not monotonic.** A flush, WanGP started from its own tab, a
  restart, or a task that slips past the hold (the one window, at most a tenth of
  a second) all take `held` back to `holding`. Before a grant the turn waits; under
  a render it says so once in the console and goes on (rule 1).
* **Held is never taken back by refusal.** A WanGP that comes up unholdable in the
  middle of a lease makes it `holding` with `bridge_hold: unsupported` until it is
  released. A turn that is still clearing refuses on that answer rather than wait
  for as long as WanGP stays up.

### 6.2 The bridge: `hold`, `resume`, `flush`

Three operations on the existing authenticated control plane, advertised by a
capability flag so an older bridge is recognised rather than refused:

* `hold {lease, ttl_s}`: set WanGP's between-task pause flag and re-assert it
  while held (WanGP's own edit feature clears it); when no worker is running, take
  the idle claim on WanGP's GPU lock. Report `holding` until no task is running,
  then `held`. The bridge resumes by itself when the hold is not renewed within
  `ttl_s`.
* `resume {lease}`: restore what hold changed, in reverse order.
* `flush {lease, level}`: `soft` moves WanGP's weights off the card into RAM and
  empties the cache; `hard` releases the model so the next task reloads it.
  Refused while a task runs, and hard also while a run is paused between tasks
  (WanGP's own Unload refuses the same way) — which is why the turn asks for soft
  first.

`hello` gains `worker`, `task_running`, `active_client_id`, `queue_length`,
`hold`, `waiter`, `vram_free` and `vram_total`. The bridge version becomes 1.12.0
(1.11.0 went to the focus-mode layout, Mini Paint NEO #110), and
`compatibility.py` remains the only file that knows a WanGP internal.

### 6.3 What changes in Mini Paint's own rules

The bridge still never starts a second run, never aborts one and never presses
Generate; it can now *hold* WanGP between tasks at Forge's request, which its
README says in as many words. The executor's "submit anyway after 60 s" rule does
not apply to a lease. `presence.report()`'s closed key set is unchanged; the lease
is a separate function.

---

## 7. Rules this repository declared, and how they change

* **Voice is CPU-only** (`README.md` "On the machine"; invariants I-9 and
  I-PKT-7). Still true of Kokoro, Sopro, PocketTTS and the cleanup engine.
  VibeVoice is the first engine that joins the broker, appears in the residency
  planner and waits for image jobs, and the README says so where it states the
  rule.
* **Voice modules do not import the memory side** (I-3,
  `tests/test_voice_independence.py`). Kept: section 4.6.
* **One TTS worker at a time** (`mc_voice_engines.select`; invariants I-1 and
  I-PKT-1). Voice Chat still has one selected engine. The Voice Box runtime is a
  workspace runtime outside that selector, one worker per card; when VibeVoice is
  also Voice Chat's engine, the two share it.
* **WanGP's VRAM is never taken** (handoff 28): section 4.4's exception.
* **An image residency is never demoted for another family** (`mc_broker`):
  a VibeVoice turn is the second door out of that rule, after the user's own LLM
  priority, and it is opened by a request the user made; the model comes back.

---

## 8. The Voice Box page

**Input.** Pick a local audio or video file. The browser plays it and draws its
waveform from the file on the user's device — nothing is uploaded to look at it.
Drag in and out points, play the selection, loop it, name it, and *Save as
sample*: only that trimmed selection is encoded (16-bit WAV) and sent. Short
files are decoded whole with `decodeAudioData`; long ones are captured by playing
the selection through an AudioWorklet, so a 20-second sample takes 20 seconds.
The server normalises the sample (`mc_voice_reference`) and adds it to the
**sample library**: a list of names with waveforms, like the output list.

**Prompt.** The script, with `Speaker 1:` … `Speaker 4:` lines mapped to samples,
optional `[pause]` and `[pause:ms]` tags (which insert silence between separately
generated sections, and say so), a history of the last prompts, and favourites.

**Configuration.** Model, card, precision, attention, diffusion steps, CFG, seed,
sampling (temperature, top-p), speaking speed, LoRA later. Saved and managed as
named configurations. Controls a model cannot honour are absent, and values shown
after a render are the ones the worker reports.

**Outputs.** A list of named renders, each a lane with its waveform, a play and a
loop control, *Trim to sample*, *Save* (to a folder on the Forge PC, chosen once
and remembered) and *Download*. Each carries its metadata.

**Pipelines.** Several at once, each a file of its own like a conversation: its
inputs, prompt, configuration reference and outputs. Renders from any pipeline
join the queue of their card.

**Audio focus.** A single event on `document`, `mc:audio-focus`,
`{owner, kind}`: Voice Box playback stops Voice Chat's speech and closes any open
microphone (the composer's, the flyout's dictation and the clone recorder's),
and Voice Chat speaking or listening pauses Voice Box playback.

**The page** is one `gr.HTML` root painted once in an `on_ui_tabs` tab, a script
bundle that owns its DOM, and JSON routes on the page token, with Mini Paint's
Gradio 4.40 traps respected throughout: no other tab's component in an outputs
list, state fetched on boot, every request with a deadline, and render progress
read by bounded polling of a server-owned record, never an open stream.

Settings that a live page changes (the save folder, configurations, favourites)
live in Voice Box's own files under the voice data root, so Forge's *Apply
settings* cannot write a stale copy over them.

---

## 9. Voice Chat

Later (phase 3): VibeVoice as a fourth engine for spoken replies, completed-reply
first. A reply waits its turn on the card like any other request, so on a card
that is busy rendering or generating, a spoken reply starts when that work ends.

---

## 10. Phases

| Phase | Delivers |
|---|---|
| **1a** | This document; the per-card turn system (`mc_turns`), the gate on txt2img and img2img, parking and return of the image model with its setting, the keep-warm setting, the voice family in the broker, the LLM's side of a turn; Mini Paint's lease, executor gate and bridge 1.12.0 |
| 1b | The VibeVoice runtime: closure and installer (mirror + folder, pinned pair of torch wheels, tokenizer), worker with a handshake that proves its card, registration as a guest, single-speaker synthesis in a first Voice Box |
| 2 | The Voice Box page: samples and local trim, prompts with history and favourites, configurations, outputs, pipelines, audio focus, up to four speakers |
| 3 | VibeVoice as a Voice Chat engine |
| 4 | Realtime 0.5B, quantised 7B, LoRA, speaking while the LLM writes |

Phase 1a has no guest in it: until 1b registers VibeVoice, the gate is a
dictionary lookup and nothing changes for anybody. It is built first because it
is the part every later phase stands on, and the part that can be tested without
a GPU.

## 11. Risks and open items

* The hold in WanGP uses WanGP internals (`queue_paused_for_edit`,
  `process_locks`), not plugin API; they are checked functionally, and without them
  the bridge reports `unsupported`.
* On the 3090 the 7B fits only with the image model fully parked: about 19–20 GB
  at peak against about 22.7 GB obtainable. The first real render measures it.
* System RAM is the scarcest resource on this machine. The margin in section 5 is
  a first value and is logged every time it decides something.
* Work that never passes the gate — Extras, other extensions' pipelines — can run
  beside a *warm* VibeVoice, because the queue lock is given back when a render
  ends. An upscale on the 3090 with 18 GB of warm guest and the image model parked
  could run short; the first report of one decides whether a warm stay should
  keep the lock after all.
* Two handoffs disagree about this machine's RAM (48 GB and 96 GB). Every RAM
  decision logs its figures, so the first turn settles it.
* Nothing in phase 1a has run on the user's machine.
* Exclusive residency (*Free the LLM for every image*) sweeps a warm VibeVoice
  off the image card with the language model, because the mode is a promise that
  the image family owns that card. The gate evicts a warm guest before every
  generation anyway, so this only shows for image work that asks the broker
  without passing the gate.
