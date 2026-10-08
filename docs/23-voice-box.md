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

**Attention.** SDPA by default, and since the fourth round a choice per
configuration: SDPA, Eager, or Flash attention 2 when its package is in the
runtime (§8, the fourth round). Microsoft's demo calls flash-attention the only
fully tested path, but flash-attention has no official Windows wheels, the three
compute the same attention up to rounding, and nobody has measured a difference.

**Solver.** The model's own scheduler, DPM-Solver++ second order, which upstream's
`inference_from_file.py` renders with; upstream's Gradio demo reconfigures the
same scheduler to its SDE variant the moment it loads. Both, and the other orders
the class finishes, are a choice per configuration since the fourth round.

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
join is `mc_turns_guests.py`, which is not a voice module either: it registers
`VibeVoiceGuest` (three methods that reach `mc_voice_vibevoice_runtime` per
card) with the turn system, and hands `mc_voice_box` a `TurnClient` through which
a render asks for its card, reads the cards' snapshot, lists the machine's cards
with their roles, and unloads a warm guest by hand — the direction of dependency
that keeps invariant I-3 (`tests/test_voice_independence.py`, where `mc_turns`
and `mc_turns_guests` are now on the forbidden list) true for every voice
module. `scripts/model_chain.py` calls `mc_turns_guests.install` from
`on_app_started`.

The Voice Box itself is four voice modules and a worker: `mc_voice_vibevoice.py`
(manifest, installer, settings, the VRAM and RAM estimates and their
calibration), `mc_voice_vibevoice_runtime.py` (one worker process per card, the
handshake that proves the card, render, cancel, evict), `vibevoice_worker/
worker.py` (the process: torch, transformers and the community `vibevoice`
package, imported lazily), `mc_voice_box.py` (samples, prompts, configurations,
pipelines, outputs, the script parser and the render service), `mc_voice_box_api.py`
(the routes) and `mc_voice_box_ui.py` with `javascript/voice_box.js` (the page).

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
and Voice Chat speaking or listening pauses Voice Box playback. Since 8 October
2026 LLM Studio's *Play VibeVoice* claims it too (`llm-studio`): Voice Box and
LLM Studio yield to any owner that is not themselves, Voice Chat to the two it
knows (`FOCUS_RIVALS`), so a stranger's event cannot cut a reply off.

**Other tabs, since 8 October 2026.** The page exposes a bridge for a script of
another tab, `window.mcVoiceBox`: `canRender(text)` — the sentence Render would
be refused with, or empty; `renderText(text, {name, origin})` — a render with the
page's own samples, speakers and configuration and the text as the script,
queued (a Promise of the job); `jobById(id)` — the job as the page's own poll
last saw it, so a caller watching a job opens no request of its own;
`outputsFor(prefix)` — the outputs whose `render.origin.key` starts with
`prefix` (a Promise); `outputAudioUrl(id)` — a blob URL of the output's audio,
fetched with the token. The render's **origin** is `mc_voice_box.render(…,
origin=)`: `{kind, key, label}`, three strings of at most `ORIGIN_CHARS`
(`_clean_origin`; anything else dropped), kept on the job (`Job.origin`,
`to_dict`) and written into the output's record (`render.origin`), changing
nothing about the render itself. It is how the asking tab finds its render after
a reload. LLM Studio's *Send to VibeVoice* is the first caller (docs/07 §39):
`kind: "llm"`, the message's key, the label *LLM Studio*; the lane is named
*LLM Studio · <the first words>*. Nothing in the Voice Box depends on who calls.

**The page** is one `gr.HTML` root painted once in an `on_ui_tabs` tab, a script
bundle that owns its DOM, and JSON routes on the page token, with Mini Paint's
Gradio 4.40 traps respected throughout: no other tab's component in an outputs
list, state fetched on boot, every request with a deadline, and render progress
read by bounded polling of a server-owned record, never an open stream.

Settings that a live page changes (the save folder, configurations, favourites)
live in Voice Box's own files under the voice data root, so Forge's *Apply
settings* cannot write a stale copy over them.

**As built (phases 1b and 2).** A sample is three to sixty seconds, normalised
to mono 16-bit at 24 kHz by `mc_voice_reference`, at most 64 MB on the way in,
and carries 240 waveform peaks so lists draw without decoding. `[pause]` is
700 ms, `[pause:ms]` is clamped to 50–10000 and adjacent pauses add up to that
cap; each section is rendered on its own and the silence put between them. The
history keeps 100 prompts and never drops a favourite. A configuration holds the
model, the card, steps 1–50 (10), CFG 1.0–3.0 (1.3), a seed or none, max new
tokens or none, and four speaker slots. A render job is `queued → waiting →
loading → rendering → done | failed | cancelled`; everything that can be refused
before a card is asked for is refused at the press; a blocked turn fails the job
with the turn's own warning; a cancel withdraws the turn of a waiting job and
tells the worker of a rendering one to stop; the card is handed back in every
ending, and the warm stay follows Settings → Model Chain (Voice Box's own toggle
can only decline it). Renders on one card run in order, one per card. Outputs are
named after their pipeline and number, saved into the remembered folder with a
JSON sidecar and never overwritten, and downloaded through the token-checked
route. Files live under `<voice data root>/voice_box/`. The routes are under
`/model-chain/voice-box` on Voice Chat's page token.

**The UX round (after phase 2, at the user's request).** Six changes, each the
user's own words first:

1. *"Nothing should render taller than browser view and force a scroll of the
   entire page."* The root takes the height the window has below it (window
   height, less the root's top, less whatever the document lays out under it,
   at least 420 px), measured and written in pixels — a percentage inside
   Gradio's containers resolves to auto — on resize, on the visual viewport's
   resize, on a tab switch and after fonts and data paint; never by observing
   the box it sizes. Every long list scrolls inside its stage.
2. *"On mobile … swipe up to get to next stage, swipe down to previous."* Under
   900 px of root width the layout is `stack` (`data-layout` on the root): the
   stages container scroll-snaps vertically, each stage the container's full
   height and width, `scroll-snap-stop: always`, scroll chaining left on.
3. *"Options are not shown until selected … the current waveform size is great,
   I just want that alone with compact date and name."* An unselected lane is its
   tint edge, its waveform at the size it had, and one line of date and name; a
   press that is not a drag selects it (one at a time, kept across polls by id,
   Enter or Space from the keyboard, `aria-expanded`), and the selected lane has
   everything a lane had plus the infotext, *Use seed* and *Reuse settings*. A
   render the page started is selected when it lands.
4. *"A seed … available for all output … an infotext sort of like a fingerprint
   … the ability to reload its configuration and prompt"*, and *"smaller files …
   maybe some .mp3 type"*. A blank seed is drawn when the job is queued
   (`secrets.randbelow`, 0 to 2³¹−1), sent to every section, and recorded with
   `seed_drawn`; the output's `render.configuration` is the configuration as the
   render used it — the saved one's id even when the render ran on unsaved
   changes to it, the seed used, every speaker slot — for *Reuse settings*. The
   infotext is WebUI-shaped (the prompt, then `Steps: …, CFG scale: …, Seed: …,
   Model: …, Speaker n: …, Max new tokens: …, Sections: …, Length: …, Render
   time: …`, a value quoted only when it holds a comma, colon, quote or newline),
   computed from the record whenever an output is handed out and never stored,
   and written into the file. Outputs are MP3 — 128 kb/s CBR mono at 24 kHz,
   encoded in the Forge process by PyAV (Forge Neo ships `av`; nothing is added
   to its environment) with the infotext as the ID3 comment and the record as
   JSON in a `voicebox` TXXX frame, and no title, which a rename would leave
   stale. LAME's delay and padding are in the file's info header, so a decoder
   gives back exactly the rendered samples. Where PyAV cannot encode, the output
   is a WAV with an `INFO` chunk (`ICMT`, `ISFT`) after its sound; renders from
   before are WAVs and are listed, played and saved as such (`format`). The card
   is handed back before the file is encoded.
5. *"Some sort of transparent light touch color applied from input file to all
   output that use them."* A hue from a stable hash of the sample's id at a fixed
   low saturation (one lightness for the light theme, one for the dark): a 4 px
   edge on the sample's row, and on every output an edge of equal stripes, one
   per sample it used, in speaker order. Decorative only.
6. *"Move the render button and queue to the Outputs. Make the queue just a
   counter … installing and warmed up status."* The footer is gone; Render sits
   in the Outputs header with one status line whose first applicable state is
   shown: an install; the running job (phase, section, elapsed from the server's
   `elapsed`, `· n queued`) with Cancel and Clear queue (`/jobs/clear`, every
   queued job on every card, the running one left to Cancel); only queued jobs;
   the last failure of a render this page started; VibeVoice warm on a card, with
   Unload; why Render cannot run; *Ready*. A fresh message (an error, *Saved
   to …*, *Copied*) takes the line first, 5 s for information and 12 s for a
   warning, with a ×; the old cards line is the line's tooltip. The queue is a
   count, so queued jobs are withdrawn together by Clear queue, not one by one;
   the cancel route still takes any job.

Fixed in the same round: the worker freed a section's render slot only after
writing its reply, so a parent that asked for the next section on reading it
could be refused "one render at a time"; the 7B's `capped` never fired (upstream's
loop ends one step before its own check); the protocol now runs on a duplicate of
descriptor 1 with standard output pointed at the log; the Voice Box noted every
render's peak a second time; and two renders queued as one ended could share a
name.

**The second UX round (after #240, at the user's request).**

1. *"Let's move it to the configuration stage."* Render, Install and the
   status line (unchanged in behaviour) head the Configuration stage; Outputs
   is its title, the lanes and the empty-pipeline sentence.
2. *"We need an explicit 'sampling' control, and when off, it disables top-p and
   temperature."* A configuration has `sampling` (off by default), `temperature`
   (0.1–2.0) and `top_p` (0.05–1.0), both 0.95. Off, the worker hands `generate`
   `{"do_sample": false}` as upstream does; on, `{"do_sample": true, "temperature",
   "top_p"}`. The model's language part only chooses control tokens — keep
   speaking, end a stretch of speech, stop — while the voice comes from the
   diffusion head, so sampling varies the pacing, never the timbre, and the seed
   still reproduces a sampled take. Only a real `true` turns it on, in the
   configuration and in the worker; while it is off the two values are kept for
   turning it on again, and one out of range is brought into range rather than
   refused, because the page greys the field out. Each output records
   `render.sampling`, `render.temperature`, `render.top_p` (null when greedy) and the
   configuration's own three; the infotext gains `Temperature, Top-p` after the seed
   when the render sampled.
3. *"Horizontal card scrolling makes a lot of sense."* Under 900 px the stages are
   cards in a horizontal scroll-snap container (x mandatory, one full card at a
   time) with a stage bar (*Input · Prompt · Config · Outputs*, `aria-current`,
   smooth or, under reduced motion, instant); each card is the one vertical
   scroller (its header outside the scrolling body, not `sticky`, because the
   positioned rows would paint over a sticky header without a z-index), lists have
   no scroll areas of their own there and the script box grows. This replaces the
   first round's vertical snap, whose inner lists took the swipe that should have
   changed stage.
4. *"That scrubbing should be avoidable for all not-playing audio."* One active
   player — the one last started and not stopped, playing or paused by its own
   Pause — seeks on a press or a drag, draws a playhead and, on a touch screen,
   keeps sideways drags (`touch-action: pan-y`); every other waveform ignores them.
   A finger's press waits until it moves along the waveform or lifts, so a
   vertical scroll that starts there scrolls.
5. *"Buttons with icons … start from the beginning, and another for STOP."* Every
   player (sample rows, the trimmer, the selected lane): Play/Pause, Play from the
   start (the trimmer's selection start), Stop (pause, back to the beginning, no
   longer active), Loop where it was; inline SVG in `currentColor`, with
   `aria-label` and `title`.

**The third UX round (after #241, at the user's request).** The user's own Forge
runs the Lobe theme, and two of its habits were behind three of the four asks.

1. *"There appears to be no way to enable sampling … I need a simple sampling
   toggle. Also, remove that description."* The theme draws every native
   checkbox itself, a fixed square with `flex: 0` and `appearance: none`, and
   against the section's own `min-width: 0` on its field inputs that left the
   box its 2 px border: it worked when that sliver was hit, and nothing said it
   was a control. Sampling, and *Keep VibeVoice warm between renders* (the same
   checkbox, the same sliver), are now switches: one button each with a drawn
   track and knob beside the name, `role="switch"` with `aria-checked`, the knob
   to the right on a filled track when on, so the side says it without the
   colour. The hint under Sampling is gone, and so is the paragraph of script
   syntax under the script box ("I don't need descriptions"): the empty box's
   placeholder is an example script. The tab has no native checkbox left.
   Sampling still marks the configuration unsaved, is saved and sent inline and
   is applied by Reuse settings; Keep warm still saves at once, and a poll that
   arrives while its save is on the way does not flip it back.
2. *"Remove that floating X."* The theme's *SVG icon* option, on by default, runs
   once when its app mounts and replaces the whole content of every `<span>` whose
   text contains `×` with a 36 px X. The status line's actions were a span holding
   the dismiss button's `×`, so on the user's page Cancel, Clear queue, Unload and
   the dismiss were one large X beside *Ready*, and the page's references pointed
   at buttons no longer in the document. The actions are a `div` now and the
   dismiss draws its cross. The section's `#mc-voice-box [hidden] { display: none
   !important }`, there since phase 2 because any author rule that sets `display`
   beats the browser's own rule for `hidden`, was not the cause; it now covers the
   root as well.
3. *"The settings manager needs to be more compact."* The configuration select,
   Save, Save as and Delete are one row at every width the stage takes: the select
   takes what is left and ellipsises, the three are square icon buttons (a disk, a
   disk with a plus, a bin), and unsaved changes are an accent dot on Save, whose
   name then says so.
4. *"If I tap anywhere outside of the selected waveform, it should deselect."* A
   click outside the selected lane's box — the whole open lane counts as inside — or
   Escape collapses it. The page listens for `click`, which a scroll or a card
   swipe never makes; a click that ends a drag, a click the page made itself and a
   click while the tab is not on screen are not taps. Playback is untouched: the
   active player plays on, keeps its playhead on the compact waveform, and a poll
   does not reopen the lane.

**The fourth round (after #243, at the user's request).** After an analysis of
how the Voice Box compares with upstream's own settings, the user asked to choose
the solver and the attention, to render several takes at once, for the best MP3
the model's sound allows, and for twelve steps by default.

1. *"Allow me to choose solver … I should see all supported options."* A
   **Solver** list: every configuration of VibeVoice's own scheduler
   (`vibevoice.schedule.dpm_solver.DPMSolverMultistepScheduler`) that finishes on
   the model's cosine noise schedule — DPM++ 2M (the model's own, and
   `inference_from_file.py`'s; the default), DPM++ 2M SDE (the Gradio demo's),
   DPM++ 3M, DPM++ 1M (DDIM) and DPM++ 1M SDE. Never another class: the model
   walks the timesteps itself with no input scaling. The deprecated algorithms are
   out, and there is no third-order SDE (that update takes no noise). The class's
   Karras and Lu spacings were built and then taken out again, before anything
   shipped: run against upstream's own code they put several steps on timestep
   999 of the cosine schedule and the scheduler runs off the end of its noise
   levels on the last step, at 45 and 42 of the step counts from 1 to 50. A
   spacing the model's own scheduler cannot finish is not an option, so there is
   no Schedule control. The worker keeps the model's scheduler object and hands it
   back for DPM++ 2M; any other is its configuration with the algorithm and order
   changed, as the demo does it.
2. *"Allow me to choose attention."* An **Attention** list: SDPA (the default),
   Eager and Flash attention 2 — the three VibeVoice declares; flex attention is
   not declared and needs Triton. transformers 4.51 reads the language model's
   `_attn_implementation` at every forward pass, so a render switches it with one
   attribute and no second load. Flash attention needs the `flash-attn` package,
   which the runtime does not install: the installer reads the runtime's
   site-packages for it, the page lists it greyed with *not installed* until it is
   there, Render refuses a configuration that holds it, and the worker refuses
   again by import before it switches.
3. *"Add option to batch up to 4 … the first in batch is 9990 and last is 9993."*
   A **Batch** of one to four takes, each its own output, take *k* at seed + *k*.
   A batch is one `generate` over the prompt repeated — the weights are read once
   for every take — and upstream draws its randomness from Torch's global
   generators in three places (the voice prompt's encoding samples a latent around
   its mean; the diffusion head's starting noise and the SDE solver's per-step
   noise; the token choice when sampling), so the worker gives each take generators
   of its own, seeded with its seed, and draws for it exactly what a render of that
   seed alone draws, in the same order and shapes (`TakeRandomness`). Which takes
   are speaking at a frame is `generate`'s own `diffusion_indices`, read from its
   frame, because upstream passes the diffusion head only their conditions.
   Upstream's streamer contract also had to be met differently: `generate` leaves
   its loop the moment *any* streamer flag is up, so a take that ends raises none,
   or it would cut the others off mid-sentence. Checked against upstream's own
   code with a tiny VibeVoice on the CPU (`tests/test_vibevoice_upstream.py`): a
   single take through the per-take draws is bit for bit the plain render, for
   every solver, greedy and sampled, one voice and two; take *k* of a batch has
   the length, the token count and the samples of the render of seed + *k* (to
   6e-9 in float32, exactly in bfloat16 there); takes that end at different steps
   do not end each other. A single take keeps the global generators, so seeds
   recorded before batches existed make the same renders. A drawn seed leaves room
   for the whole batch below the largest seed; a fixed one that would pass it is
   refused at Render. The turn asks for the weights once and a working set per
   take, and the calibration keeps one take's peak.
4. *"I want the output to be .mp3 … high fidelity … go straight to high fidelity
   .mp3."* The worker answers with the model's 32-bit float samples (protocol 2),
   and the MP3 is made from them in memory: 160 kb/s, the ceiling of MPEG-2 Layer
   III and so of an MP3 at the model's 24 kHz, at LAME's quality 0, constant
   bitrate so seeks land where the page asks. Mono at 24 kHz, because a second
   channel would be a copy of the one the model makes and a higher rate a
   resample: measured, LAME keeps everything to about 11.3 kHz at this rate, which
   is where the resampler that prepares every voice sample cuts, and turning its
   lowpass off changes nothing. The 16-bit copy the page's peaks and a fallback
   WAV use is scaled as the worker scaled it before. The pipe's reply ceiling is
   2 GB, four takes of the longest script as float.
5. *"Make default 12 steps."* New configurations and the engine's settings start
   at twelve; a configuration saved before keeps the count it was saved with.

*"Exposed the Sampling setting 'Greedy' … Is it exposed already?"* It is: the
Sampling switch, off by default, is greedy decoding, what upstream ships.

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
| **1b** | The VibeVoice runtime: the closure and installer (a community mirror or a folder; torch from the CUDA 12.8 index at a pinned version, no torchaudio; the Qwen tokenizer beside the weights and a local processor config that points at it), the worker with a handshake that proves its card, one worker per card, registration as a guest through `mc_turns_guests`, calibration of the estimates from every render's peak |
| **2** | The Voice Box page: samples and local trim, prompts with history and favourites, configurations, outputs with loop, save, download and trim-to-sample, pipelines, audio focus, up to four speakers, the render service and its routes |
| 3 | VibeVoice as a Voice Chat engine — built in handoff 33 and rolled back at the user's request; as first judged, not built: a spoken reply from an eighteen-gigabyte guest that has to take a turn on a card is a different latency class from the CPU engines, and the completed-reply-first design of section 9 wants a measurement of the first real renders before it is worth a fourth engine |
| 4 | Realtime 0.5B, quantised 7B, LoRA, speaking while the LLM writes — built in handoff 33 and rolled back with phase 3; as first judged, not built, for the same reason and one more: the 0.5B and the quantised weights are different checkpoints with different memory figures, and the manifest's pins for the 7B have not been made on a machine that reaches the hub yet |

Phase 1a has no guest in it: until 1b registers VibeVoice, the gate is a
dictionary lookup and nothing changes for anybody. It was built first because it
is the part every later phase stands on, and the part that can be tested without
a GPU. Phases 1b and 2 were built together, against doubles of the worker and
the cards: nothing in them has run on the user's machine, and the first real
render is the measurement every estimate here waits for.

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
* LLM Studio's *Send to VibeVoice* (docs/07 §39) has not run on the user's
  machine either: the first things to watch are the row's note when the Voice Box
  refuses (no sample for Speaker 1), the lane *LLM Studio · …* landing in the
  Outputs stage, and Play coming alive on the message within a second of it.
* Nothing in phases 1a, 1b or 2 has run on the user's machine. The first
  things to watch: the worker's handshake reporting the card asked for; the
  7B's real peak on the 3090 against the estimate (the calibration takes it from
  there); the installer against the mirror's actual shard list; and the trimmer
  on a real video file in LibreWolf.
* Exclusive residency (*Free the LLM for every image*) sweeps a warm VibeVoice
  off the image card with the language model, because the mode is a promise that
  the image family owns that card. The gate evicts a warm guest before every
  generation anyway, so this only shows for image work that asks the broker
  without passing the gate.
