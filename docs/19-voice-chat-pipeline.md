# Voice Chat: the Voice Pipeline, and the clock it must not break

Voice Chat's speech is as good as the model that makes it. PocketTTS on a CPU
is a small model, and small models produce speech with a narrow top end and a
faint synthesis texture under it. Two published models fix those two things:
DPDFNet takes the texture out, and LavaSR puts a top end back.

This change adds them as an optional server-side stage between the speech
engine and the browser:

```
PocketTTS → DPDFNet → LavaSR → 48 kHz mono → /tts-stream → browser → 1.0×
```

Off by default, installed separately, enabled per stage, and coupled to
PocketTTS's residency so that the models are warm exactly when there is speech
to polish and gone the moment there is not.

The user-facing rule is one line:

> **Voice Pipeline [ON] · 1 DPDFNet [ON] · 2 LavaSR [ON]**

Everything below is the machinery that makes that line safe.

---

## The one thing that is genuinely different

The other three optional things in Voice Chat — a second engine, a cloning
tool, a recording cleaner — are all *batch*. They take an input, think, and
hand back a result, and if they take a moment nobody's speech stutters.

This one sits inside a live stream that two pull requests were spent making
fast. PR #124 took the browser's startup buffer down to 0.7 s and PR #126
established that the remaining first-audio cost is source production time
rather than anything a buffer can hide. So an enhancement layer has exactly one
way to be acceptable:

> **Add algorithmic latency and inference time, and nothing else.**

No safety reservoir. No "wait until two seconds are ready". No fixed server
prebuffer in front of the browser's own adaptive one. The earlier draft of this
design proposed 1.5 s of processed audio before playback; that number is not in
this implementation and putting it back would undo both PRs.

The second difference is subtler and is where the real risk is. This layer can
be *wrong about time* while sounding completely normal. A denoiser that quietly
loses twenty milliseconds per reply, or a bandwidth extender that interprets
24 kHz samples as 16 kHz, both produce audio that is unmistakably speech. One
is a little short. The other plays a third too slowly. Neither raises anything.

So most of this document, and most of the tests, are about duration.

---

## Where the pipeline begins

Not at the model. This is the single most load-bearing decision in the change.

The current PocketTTS worker does a great deal to its own output before the
parent ever sees it, and every bit of it was added on the strength of
measurements from real machines:

```
Pocket model generation chunks
    → delivery DSP            speed / pitch / gain          (Shaper)
    → trim                    generated quiet, normalised    (#149, #150, #151)
    → internal-gap policy     long pauses inside a unit      (#152)
    → seam                    8 ms raised-cosine unit edges  (#148)
    → intentional pause
    ================= VOICE PIPELINE INPUT =================
```

The pipeline consumes the PCM *after* all of that — which is to say, exactly
the bytes that used to go straight to `VoiceTurn.offer_audio`. In the code that
is one branch, at `mc_voice_pocket_runtime._dispatch_turn`:

```python
if operation == "tts_audio":
    enhancing = getattr(turn, "pipeline", None)
    if enhancing is None:
        turn.offer_audio(payload, int(header.get("sample_rate") or 0))
    else:
        enhancing.offer(payload, int(header.get("sample_rate") or 0))
```

Inserting before those operations instead would have meant enhancing audio the
source intends to throw away, paying inference time for it, disagreeing with
the source's own clock, and running two seam systems against each other.

### The corollary somebody will want to "fix"

If the trim removed 300 ms of dead air, that 300 ms is **not** in the
pipeline's input and does not come back. Preserving duration means preserving
the finalised clock this layer was handed — not the model's hypothetical
pre-trim one. A pipeline that restored it would be a pipeline undoing four
merged PRs from the inside.

---

## The clock

If `N` samples arrive at `Fs_in` and the turn's output rate is `Fs_out`, the
committed output is exactly

```
target(N) = (N · Fs_out + Fs_in // 2) // Fs_in
```

computed from the **running total**, never per packet. For today's path — 24 kHz
in, 48 kHz out — that is exactly `2N`, and the general form is kept anyway so
that the first engine to speak at 22050 does not find a special case waiting for
it.

Each stage owns its half:

| stage | rate | contract |
| --- | --- | --- |
| DPDFNet | caller's, unchanged | every sample in comes back out; its latency is a debt paid at the flush |
| LavaSR | 48 kHz out | `target()` of what it consumed, reconciled at the end of the turn |

The reconciliation is counted rather than performed quietly. A well-behaved
backend needs none of it; the counter is what a wrong clock shows up as, and
section 26.17 is explicit that a large correction means the adapter is wrong and
must not be normalised away.

---

## The rate question, answered

Upstream's README advertises 8–48 kHz input and a 24→48 kHz evaluation. Its
code says something else. `LavaSR/model.py`:

```python
if denoise:
    wav = self.denoiser_model.infer(wav)
    wav = torchaudio.functional.resample(wav, 16000, 48000)
else:
    wav = torchaudio.functional.resample(wav, 16000, 48000)
```

**Unconditional, in both branches.** Whatever `enhance()` is handed is
interpreted as 16 kHz. `load_audio()` will load a file at some other
`input_sr`, which is what makes the README plausible, but the inference path
does not care what you loaded — hand it Pocket's 24 kHz and the reply comes back
half again too long, sounding like speech, with nothing raising anything.

So the contract is settled and recorded in the manifest: **`backend_input_rate:
16000`**. The adapter resamples the real source rate to 16 kHz deterministically,
runs the model, and takes 48 kHz out — which is exactly what `LavaStage` already
does with `Resampler(rate_in, backend_rate)`.

That was the blocking unknown. What still blocks *LavaSR* is something else
entirely, and it is not about audio.

### What each stage's state actually is

| | DPDFNet | LavaSR |
| --- | --- | --- |
| Rate contract | measured (caller's rate, both ways) | measured (16 kHz in, 48 kHz out) |
| Package | `dpdfnet` 0.6.0 on PyPI | none published |
| Dependencies | pinnable: onnxruntime, librosa, numpy | Torch, torchaudio, and a `vocos` **fork on a git branch** |
| Model | HF `Ceva-IP/DPDFNet`, resolved at install | HF `YatharthS/LavaSR` |
| **Installable** | **yes** | no |

DPDFNet installs and runs. Its closure is pinned byte for byte — 32 wheels
resolved from PyPI for `windows-x86_64-cp313`, each one hashed and checked
against the digest pypi.org publishes for it — and the streaming path is
upstream's own `StreamEnhancer`, handed an explicit `onnx_path` so that
`resolve_model()` short-circuits every search and every download.

LavaSR does not, and the reason is a dependency closure rather than a doubt
about the audio: there is no wheel, and `vocos @ git+https://…@matcha` is a
branch of a fork, which is not something this repository can pin the way it pins
everything else. Its adapter is written and tested against stand-in backends;
what is missing is something a release could stand behind.

### Provisional pins, and why they are not a loophole

The DPDFNet **model** is declared but not hashed: the machine that wrote the
manifest could not reach huggingface.co, so its revision is the publisher's
branch and its digest is whatever the publisher reports at download time — which
the installed record then keeps.

That is weaker than a committed hash and the difference is worth stating: both
refuse a file that arrives wrong, only one refuses a publisher who changed their
mind. So a stage in that state carries `"provisional": true`, and everything
downstream says so — the settings row, the status line, the installed record.
`_read_stage` refuses a branch revision on any stage that does *not* declare it.

To turn it into a real pin, from a machine that can reach the hub:

```
python tools/pin_pipeline_models.py --stage dpdfnet --revision <sha>
```

### Still outstanding for a release

1. Sweep 250 / 500 / 750 / 1000 ms Lava analysis windows with 40–120 ms context
   and choose the **smallest** acceptable — Voice Chat spent two PRs taking
   latency out and this feature does not get to put it back by default.
2. Benchmark DPDFNet + Pocket Maximum concurrently for sustained RTF on
   reference hardware.
3. Replace the provisional DPDFNet model pin with an immutable revision.
4. A pinnable LavaSR closure, or a decision to vendor its inference.

---

## The rolling window, and why it is not a packet converter

DPDFNet is genuinely streaming: arbitrary caller chunk sizes, a 20 ms first
window, ~10 ms hops. It needs an adapter that buffers to its block size and
tracks what it owes.

LavaSR is not. It is given an analysis window with context on both sides, and
`concat(lava(packet_0), lava(packet_1), …)` would put a hard join at every
packet boundary. So:

```
window i reads   [i·H − C, i·H + A + C)     A = analysis, C = context
window i keeps   [i·H,     i·H + A)
window i commits [i·H,     i·H + H)         H = A − O
window i carries [i·H + H, i·H + A)         into window i+1's head
```

The committed stream advances one hop per window, and every place where two
windows meet is a ramp rather than a join. Only the overlap region itself is
produced twice — `O` of every `H` samples — which is the point of crossfading
rather than concatenating.

The ramp is **amplitude-complementary** (raised cosine summing to one), not
equal-power, and that is a measurement rather than a preference: the two things
being faded are two model outputs of the same audio, so they are almost
perfectly correlated, and an equal-power pair would sum to 1.41× in the middle —
a 3 dB bump at the analysis-window cadence, which is exactly the periodic
pumping the test plan says to reject.

This window seam is **separate from** the source worker's 8 ms unit seam. That
one fixes the edges of a spoken unit; this one fixes the edges of an analysis
window, which exists only inside the enhancement worker and which the source has
never heard of. Neither is allowed to do the other's job.

### Packets and units mean nothing here

Transport blocks are framing. Synthesis units are the source engine's business.
Neither resets a stage, flushes a window, or changes a duration. The same PCM
delivered as one packet and as 250 one-sample packets produces **bit-identical**
output, which `TestTransportChunksHaveNoMeaning` asserts rather than
approximates — nothing in either path is packet-aware, so there is no
floating-point reason for them to differ and a tolerance would hide the bug the
test is for.

---

## One queue, and it is not a prebuffer

```
Pocket worker → parent reader thread
                    ├── turn is draining ────────→ read and dropped
                    └── turn is speaking
                            ├── pipeline off ────→ VoiceTurn.offer_audio
                            └── bounded ingress → pump → worker → VoiceTurn.offer_audio
```

The ingress holds at most ~2 s of source PCM. **Nothing waits for it to fill.**
It is emptied as fast as the worker will take it, and the first finalised sample
goes straight to the VoiceTurn. What the bound is for is the other direction: if
enhancement falls behind production for two seconds, the pump stops taking
blocks, the reader thread waits, the pipe behind it fills, and the source slows
down. That is a real-time-factor problem being made visible instead of hidden in
a buffer that grows until something runs out of memory.

The browser keeps its own policy — 0.7 s start, 0.4 s floor, 2.0 s ceiling,
1.0 s rebuffer — untouched.

### Ending a turn deliberately waits for nothing

There is one thread reading the enhancement worker, and it does two jobs:
delivering audio into the playback queue, and carrying replies back to whoever
asked a question. Delivering audio *blocks* when the listener's buffer is full —
which is not an anomaly but the designed backpressure, and on a phone whose page
went to the background it can last minutes.

So ending a turn asks nothing. `turn_end` goes down the pipe fire-and-forget,
and `turn_flushed` comes back as an ordinary frame **behind** every remaining
block of that turn's audio. Reading it in order is all that was ever needed: it
cannot arrive before the reply's last sample has been handed to playback, which
is exactly the guarantee a request/reply wait was trying to buy — and could not,
because the thread it would have waited on was the thread delivering the audio
it was waiting for.

That is not a hypothetical. With a wait there, a listener who stopped draining
for thirty seconds got a reply the worker had processed perfectly, truncated,
with an error banner.

What *is* bounded is inactivity, not elapsed time. The flush watchdog gives up
only when nothing has moved — no sample delivered, and the reader not parked in
playback — for a minute, which distinguishes a listener who is not listening
from a worker that has stopped answering. Those are two different waits and only
one of them can honestly be given a deadline.

### The drain branch is the reason the insertion point is where it is

Look at the diagram again. A draining Pocket turn's frames are read and dropped
**before** anything downstream sees them. That is not a check this feature
added; it is where `_dispatch_turn` already dropped them, and inserting after it
is what makes the invariant structural:

> A cancelled Pocket unit, which must be allowed to finish inside the model,
> can never block on the enhancement queue — because its frames never arrive at
> it.

`TestTheCancelledDrainIsNeverBackpressured` asserts the branch ordering itself,
because that ordering *is* the guarantee.

---

## Residency

```
Pocket loads      → pipeline warms beside it (overlapped cold start)
Pocket resident   → pipeline resident
Pocket unloads    → pipeline unloads
WebUI exits       → pipeline dies, gracefully or otherwise
```

There is deliberately **no idle timer**. The recording cleaner has one because
it runs while somebody is tidying a clip and should not hold a Torch runtime for
an afternoon afterwards. This one runs while somebody is having a conversation:
a timer would make it cold exactly when the next reply arrives, and resident
with nothing to enhance the rest of the time. Residency follows an engine, never
a clock.

Containment is the same tested pair every other worker has — a job object with
kill-on-close on Windows, proved at the parent with `IsProcessInJob`; and
`PR_SET_PDEATHSIG` inside the child on Linux, with the parent re-checked after
setting it. An untested third mechanism would be a promise this feature has no
way to keep, so unsupported platforms refuse to start a process at all.

---

## Failure, and what it costs

| when | what happens |
| --- | --- |
| before the first byte | the reply is spoken **unenhanced**; the format has not been committed to, so this is honest |
| after bytes have been played | the turn ends cleanly and the next one recovers |

There is no third option after the stream has started. Splicing unenhanced
24 kHz into a 48 kHz response would change the rate mid-reply; continuing with
the last processed block would be a stutter presented as speech. Both are worse
than a reply that stops.

A stage the user enabled but never installed is **named**, not silently skipped:

> DPDFNet is switched on but not installed, so it was left out of this reply.

and the path preview is generated from the turn snapshot rather than from the
switches, so it can never claim a stage ran that did not.

---

## Trust

Everything the repository already does, kept:

- checked-in manifest is the trust root; artifacts are pinned by revision, byte
  count and SHA-256, and `main` is refused as a release identity;
- the runtime closure ID is **derived** from the pinned artifacts, so re-pinning
  a wheel makes every installation stale by arithmetic rather than by somebody
  remembering to bump a number;
- installs stage into a temporary tree, verify, self-test, and only then
  promote — a failure anywhere leaves the machine exactly as it was;
- the enhancement runtime is separate from PocketTTS's. Lifecycle coupling does
  not imply dependency coupling: Pocket's closure stays reproducible, the
  pipeline can later serve Kokoro and Sopro, and a pipeline update cannot mutate
  the environment a working voice depends on;
- one shared Hugging Face credential, parent-side only, moved to the **top** of
  the settings page. It is removed from the worker's environment rather than
  merely not added, because the child inherits;
- the worker has no HTTP client, no hub client, and no model name it could
  resolve. It is handed verified local directories.

The self-test the installer runs before promoting is the Phase-0 gate in
miniature: load from a local directory, feed one second of 24 kHz audio, and
refuse an installation whose answer is the wrong length or which needed a
correction nobody could call tiny.

---

## Settings

The page had grown one expanded installer block per engine, in a column, and a
fourth kind of thing — an enhancement stage, which is neither an engine nor a
utility — is where that stops scaling.

```
Access token                 [••••••••]  Saved ····abcd
─────────────────────────────────────────────────────────
Voice Pipeline                                    [ ON ]
  1  DPDFNet   Clean noise and synthesis artifacts [ ON ]
  2  LavaSR    Restore speech bandwidth to 48 kHz  [ ON ]
  PocketTTS → DPDFNet → LavaSR → 48 kHz output
─────────────────────────────────────────────────────────
Installation & Components
  TTS Engines
    PocketTTS       [Installed] [Loaded]  [Selected]
    Kokoro          [Installed] [Unloaded]
  Voice Pipeline
    Runtime         [Installed] [Loaded]
    DPDFNet         [Installed] [Loaded]  [Enabled]
    LavaSR          [Installed] [Loaded]  [Enabled]
  Voice Input / Utilities
    Recording cleanup   [Not installed] [—]
  ─────────────────────────────────────────────
  [ the one selected component's full detail ]
```

Three rules it keeps:

1. **Install state and runtime state are two chips, never one word.** "Installed"
   is a fact about disk; "Loaded" is a fact about memory right now. Reading one
   as the other is the confusion this redesign exists to end.
2. **Exactly one detail surface is mounted.** Clicking another component replaces
   the host's contents; it never appends a second panel and never leaves the
   previous component's controls in the document. Built on the surface-swap
   mechanism PR #133 already introduced, not a second refresh system.
3. **The order is numbered and immovable.** No drag handle, no move buttons, no
   order in the persisted settings and none accepted by the route. A control that
   merely *looked* draggable would be a promise the feature does not keep.

The two pipeline stages show `Loaded with PocketTTS` / `Will unload when
PocketTTS unloads` rather than a Load button of their own, because a control
that let somebody load a stage independently would contradict the residency
contract the rest of the feature is built on.

---

## What this change deliberately does not do

- change the LLM, the segmenter, or text-generation timing;
- reimplement the source engine's trim, gap policy or seam;
- restore audio the source removed;
- invent stereo width;
- change speaking speed to avoid an underrun — playback is exactly 1.0×, and a
  sustained RTF above 1.0 is a performance failure to report, not a clock to
  corrupt;
- touch the recording-cleanup DeepFilterNet subsystem, which cleans *input* and
  is a different feature;
- import a single enhancement dependency into the Forge process.

---

## Attaching another engine later

The pipeline's own contract is already generic: a turn id, PCM, a real sample
rate, a channel count and a snapshot. Nothing in the worker knows what a Pocket
unit is.

What is Pocket-only is the *plumbing* — the handoff lives in
`mc_voice_pocket_runtime` because that is where that engine's finalised PCM
exists. Kokoro and Sopro each need somebody to declare where theirs is, after
their own delivery DSP, and then they attach to the same contract.
`mc_voice_pipeline.SUPPORTED_ENGINES` is a tuple for exactly that reason: the
second entry is a plumbing change, not a design one.
