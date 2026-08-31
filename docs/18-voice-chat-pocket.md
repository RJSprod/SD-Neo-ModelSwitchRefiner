# Voice Chat: a third text-to-speech engine, and the Stop it cannot promise

Voice Chat has two text-to-speech engines and a facade that was written for
exactly two. `installed()` asked "is this Sopro?", `refusals()` asked the same
question, and `adapter()`, `runtime()`, `profiles()` and `_stop_all()` each
ended in an `else` that silently meant Kokoro. That is a shortcut while there
are two engines and a defect the moment there are three: an `else` that means
Kokoro is a Pocket voice resolved out of the wrong bank, an uninstalled engine
reporting somebody else's readiness, and a worker left running by a stop that
only knew two names.

This change ends those assumptions and adds PocketTTS behind them: a streaming
CPU model with precomputed official voices and reference-conditioned cloning,
installed separately, in a PyTorch runtime that is neither Forge's nor Sopro's.

The user-facing rule is unchanged, and everything below is the machinery that
keeps it survivable:

> **One text-to-speech engine is selected for the whole WebUI at a time.**

Speech-to-text is not in that selector and never will be. Whisper has its own
model, its own process and its own quality tier, and switching between any two
of the three does not reload it, change it or touch the microphone.

---

## The one thing that is genuinely different

Kokoro and Sopro cancel. Their synthesis is a generator the worker stops
pulling from, so abandoning it abandons the work and the engine is free at
once.

Released PocketTTS 3.0.2 is not like that. `generate_audio_stream()` runs its
own generation and decoder threads; abandoning the generator leaves the
generation thread running for the remainder of the input; and the model is
documented as **not thread-safe**, so starting the next generation while the
old one is still alive is incorrect rather than merely impolite. Upstream's own
change to add cooperative cancellation is open rather than merged, and it says
in as many words that draining the stream to completion was the correct
Python-API behaviour before it.

So Pocket's Stop does not claim to cancel. It claims what it can do:

> **Stop what I am hearing now.**

Playback goes silent immediately, on every engine, and that has not changed.
What is different on Pocket is what happens afterwards:

```
PLAYING
   |  user Stop, browser disconnect, or a superseding reply
   v
MUTED_DRAINING
   |  the browser is silent already
   |  no further text is accepted for this turn
   |  queued not-yet-started units are discarded
   |  the one native call already in flight runs to its end
   |  everything it produces is consumed and thrown away
   |  no new Pocket inference may overlap it
   v
READY
   the worker reports that its call returned
```

The Play control shows **Voice finishing…** while that lasts, with accessible
text saying that playback has already stopped and PocketTTS is finishing the
current speech unit. It clears on the worker's own report and on nothing else —
never a timer, because an indicator that clears itself is an indicator that
lets a second generation start inside a process where the first is still
running.

Kokoro and Sopro never show it. Their Stop code paths are untouched, and
routing them through Pocket's mute-and-drain to make the shared code look
uniform would be trading a correct cancellation for a workaround they do not
need.

### Muted is not "stop reading"

This is the part that is easy to get wrong and the reason the parent, the
runtime and the worker all have a line about it.

PocketTTS 3.0.2 creates ordinary **unbounded** `queue.Queue` instances between
its generation, its decoding and the generator a caller consumes. A reader that
stopped reading would therefore not stop Pocket generating. It would only leave
a producer running with nobody draining it, fill the pipe, and make the wait
*longer* rather than shorter.

So every frame is read, at every layer, and the audio is dropped at the earliest
safe point:

* the **worker** stops offering blocks onward but keeps pulling the generator to
  its end;
* the **runtime's** reader keeps consuming frames for a draining turn and
  discards the PCM at its single dispatch point;
* the **turn's** `offer_audio` returns `False` without blocking once cancelled,
  and counts what it threw away, because "the drain kept consuming" is only
  checkable if something counted it.

### Only the unit already inside the model may drain

A turn with unit 2 inside Pocket, unit 3 queued and unit 4 not yet committed
loses 3 and 4 outright. Draining a whole queued assistant answer would turn a
bounded compatibility policy into a long lockout, and the size of the unit in
flight is exactly what a Stop has to wait for — which is why the first release
sends **one committed safe unit per native call** and treats coalescing as a
measured policy rather than an optimisation.

### None of this is a fork

Model Chain does not vendor, reproduce or maintain upstream's unmerged
cancellation. When Kyutai merges it and this project deliberately adopts a
reviewed release, the worker's `INTERRUPT_MODE` becomes `cooperative`, the
interrupt sets an Event instead of running to the end, and nothing else changes:
not the `tts_interrupt` command, not the parent's state machine, not the
browser's Stop, not the busy indicator. It simply clears much sooner.

---

## The facade is a registry now

`mc_voice_engines.py` holds a table instead of a pair of branches:

```python
EngineSpec(id, label, blurb, adapter, runtime, profiles)
```

Three rows, in selector order, with the module names as **strings** imported
lazily — so importing the facade still imports no Torch, no sherpa-onnx and no
engine. Every operation below it is a lookup: `installed()` asks the adapter's
own status, `refusals()` asks the adapter which exceptions it refuses with,
`_stop_all()` walks the table, and a fourth engine is a fourth row rather than a
hunt for every place two were assumed.

The adapter contract gained three functions the facade now *asks* rather than
infers:

| function | why it exists |
| --- | --- |
| `capabilities()` | behaviour, not decoration: which routes exist, which panels are drawn, and what Stop is allowed to promise |
| `refusals()` | which exceptions are a *state* rather than a fault, so a log can tell "no voice created yet" from "the bank is broken" |
| `public_status()` | the common subset every engine answers with, so the status payload is composed rather than branched |

`resolve()` now carries `entry["_handle"]` — the engine's own address for a
voice, under the one name every engine answers to. Kokoro's is an integer
speaker in a block of floats; Sopro's and Pocket's are their own stable ids. No
shared caller may branch on which, and after this change none does.

### What Stop means, as a table

| engine | `interrupt_mode` | what it does |
| --- | --- | --- |
| Kokoro | `cancel` | synthesis is abandoned; the lane is free at once |
| Sopro V2 | `cancel` | the same |
| PocketTTS 3.0.2 | `drain_unit` | silence now; the in-flight unit finishes silently; ready shortly |
| PocketTTS, later | `cooperative` | the same user-visible behaviour, much sooner |

The browser reads that value out of the status payload. It does **not** infer
"Pocket means drain" from a version string, an engine id or anything else it
recognised, because the day a build ships with cooperative cancellation is the
day a hard-coded browser rule becomes a waiting state nobody can clear.

---

## Two states, not one boolean

Pocket's public model assets and its precomputed official voice states are in
one Hugging Face repository. Its **cloning-capable weights are a different
repository, behind an access gate** whose conditions have to be accepted
upstream with a real account.

Voice Chat cannot accept them on somebody's behalf — legally or technically —
and does not pretend to. So "Pocket speaks" and "Pocket clones" are separate
answers, and the status carries five readiness fields rather than one:

```
platform_supported   speech_model_ready    cloning_ready
runtime_ready        official_voices_ready
```

which produces the states a real installation actually passes through:

| state | what the user sees |
| --- | --- |
| Pocket selected, nothing installed | written chat works, speech is quiet, **Install PocketTTS** is offered |
| runtime + model + voices installed, gate not accepted | official voices speak; the clone panel explains what cloning needs |
| the gated weights installed too | **Clone voice** is enabled |
| interrupted on 3.0.2 | playback silent, `busy` and `draining` true, **Voice finishing…** until the worker reports ready |

A failed gated install cannot take a working installation away. The two halves
write separate markers and the cloning files are moved into place one at a time
rather than by a directory rename, so the official half is never absent even for
the length of a move.

### The credential

If a token is available to the WebUI's own process, the installer uses it to ask
the publisher and to fetch. Everything else about it is a rule:

* read from the process environment only — never a settings file, a manifest, a
  registry, a request body or `config.json`;
* attached only to the publisher host named in the artifact's URL;
* **removed before any cross-host redirect.** This one needed writing down and
  code, because `urllib` copies every header except `Content-Length` and
  `Content-Type` onto a redirected request, and a gated hub answers with a
  *signed* delivery URL on a storage host. Fetching a gated file with the
  default opener would hand somebody's Hugging Face token to a CDN as a matter
  of routine;
* never in the worker's environment. Its job is inference from verified local
  files and it has no business holding a credential for a host it will never
  contact.

A 401 or 403 becomes a `Gated` refusal, which is a sentence about access rather
than a download that failed — because telling somebody to retry the one thing
that cannot work is worse than telling them nothing.

---

## A voice state belongs to a model

A custom voice's **stable id is product identity** and is model-independent.
The safetensors state it speaks from is **model-specific**, so it lives at:

```
clones/<uuid>/
  metadata.json
  reference.wav
  states/<fingerprint>.safetensors
```

rather than as one `voice.safetensors` at the clone root. That layout prevents
two failures at once: a model switch overwriting the only state that worked with
the old model, and a state being loaded into an incompatible model because the
filename happened to exist. Switching the model back makes the old state usable
again with no rebuild.

`reference.wav` is retained on purpose. It is the only durable source a later
Pocket can rebuild a derived state from, and **Rebuild** is only a feature if
the source still exists. Deleting the voice deletes the recording.

The fingerprint covers the pinned wheel closure, the installed PocketTTS and
Torch versions, the digests of the model artifacts that were actually verified,
this build's voice-state schema, the model id — and, conservatively, the
precision. Sopro can leave quantization out of its equivalent because its INT8
misses the encoders that produce the tensors it saves; nobody has established
the same for Pocket. Until **Gate P-VOICE-1** does, a precision change marks a
cached state as needing a rebuild rather than loading one that may not mean the
same thing. The voice library then says *Rebuild*, which is a remedy, rather
than hiding the voice.

---

## Speed is ours, and the page says so

The reviewed PocketTTS generation API has no speaking-rate argument. Pocket
Speed is therefore Voice Chat's own time-scaling around the model output —
streaming SOLA with its state carried across the chunks Pocket streams at — and
Pitch is an independent resample composed on top of it:

```
time-scale asked of SOLA  = speed / pitch_ratio
resample ratio afterwards = pitch_ratio
```

At neutral both are 1.0 and neither object is constructed. A naive resample that
transposed the voice is not parity and does not ship, and the only way to know
which of the two was built is to measure it: `tests/test_voice_pocket_worker.py`
synthesizes a tone, pushes it through the shaper in awkward-sized chunks, and
checks the fundamental at every speed and the duration at every pitch.

That DSP is the same arithmetic as Sopro's and is **deliberately duplicated**
rather than shared. A shared module would have to be importable from inside
three isolated interpreters, which is exactly the coupling the separate runtimes
exist to prevent. Both copies are held to the same golden signal tests, which is
what keeps a duplicate from becoming a divergence.

The reference *decoder* went the other way. It was lifted out of
`mc_voice_sopro.py` into `mc_voice_reference.py`, because that duplication would
have drifted silently: the resampler there is band-limited with a measured 102
dB of alias rejection, and a second copy nobody swept would fail by making every
clone sound slightly worse than its reference rather than by raising anything.
What stays with each engine is the *policy* — how long a reference may be, how
loud, and what the engine is called in the sentence that refuses it.

---

## There is no thread control, and a sentence where one would be

PocketTTS 3.0.2 calls `torch.set_num_threads(1)` itself and takes its
parallelism from its own generation and decoder threads. Setting
`OMP_NUM_THREADS=8` and labelling it "8 Pocket threads" would be reporting
something that is not true, so the Engine settings panel says:

> PocketTTS manages its CPU execution policy internally in this build. There is
> no supported thread-count control.

If a later Pocket release exposes a supported thread policy, that becomes a new
*tested* engine setting. It is not anticipated by accepting arbitrary integers
now.

---

## What is engine-global and what is a character trait

| character | engine-global |
| --- | --- |
| voice | precision / quantization |
| speed, pitch, volume, pause | generation quality (sampler decode steps) |
| variation (temperature) | model / language |

A character carrying a compute setting would be a character whose turn to speak
silently restarted a subprocess and invalidated every warmed voice state. So the
character file gains six flat scalars and no more:

```
pocket_voice  pocket_speed  pocket_pitch  pocket_gain  pocket_pause  pocket_temperature
```

Flat because the character format is a deliberately small scalar-only writer
shared with oobabooga's, and a nested mapping in it would be a mapping nothing
could read back. Absent because absence is what inheritance is made of: a
character with nothing here follows Pocket's current defaults, which is what
every character written before this existed does.

`Character.voice_fields` no longer ends in a Kokoro fallthrough. An engine it
does not recognise is refused, because that branch would otherwise write a
Pocket voice id into the Kokoro field and quietly replace a character's Kokoro
voice.

---

## Failure behaviour

| situation | behaviour |
| --- | --- |
| Pocket selected, not installed | written conversation works; Auto Speak stays quiet; the panel offers Install. No Kokoro or Sopro panel appears and nothing switches back |
| runtime and model installed, cloning gate unavailable | official voices speak; the clone panel says cloning access is required. No fallback engine |
| worker fails to start | written conversation works; status names the start failure; no other TTS engine starts |
| Stop while Pocket is synthesizing | browser silent at once; no new text accepted; queued units dropped; the in-flight unit drains with its PCM discarded; **Voice finishing…** until the lane is free |
| Stop while Kokoro or Sopro is synthesizing | unchanged; no waiting state is introduced |
| browser disconnects mid-turn | the same drain policy server-side. The absence of a browser is not permission to stop draining |
| drain exceeds the bounded grace period | the worker is stopped and restarted as a failsafe, with a safe warning. Never the normal implementation of Stop |
| custom voice deleted | the character resolves to the Pocket default and the editor warns. Never to Kokoro or Sopro |
| voice state stale, reference present | the library says **Rebuild**. Conversation never silently re-encodes inside a turn |
| voice state stale, reference gone | the voice is unusable and says why. No invented state |
| engine switched while speaking or draining | the browser is silenced, the turn retired, the worker stopped. A switch is a lifecycle boundary and does not wait for the drain contract |
| stale page mutates after a switch | 409 with an active-engine mismatch flag; the browser redraws |
| publisher offers no digest | the existing trust policy applies: size checked where available, the digest of what arrived recorded, and the message does not pretend it was publisher-verified |
| reference too quiet, short or invalid | refused before any registry commit, with the actual reason |
| WebUI closed during a preview or a drain | the worker dies with the parent through all five doors; the unsaved preview is removed |

---

## What is provisional, and says so

The manifest ships **unpinned**. `voice/managed-pocket-models.json` carries the
runtime closure written down as `[package, version, kind]` triples and every
artifact list empty, with `"pinned": false`. That is a state rather than an
oversight: an artifact this repository makes no claim about is an artifact it
will not download, and the managed install therefore refuses with a sentence
naming `tools/pin_pocket_models.py`, which a maintainer runs on a machine that
can reach the publishers.

Install-from-a-folder still works in the meantime, and what is supplied has its
digests recorded and becomes the constant the next install is checked against.

Everything about the *shape* of the closure follows the Sopro precedent: exact
wheels, unpacked into an isolated interpreter without pip, self-tested before
promotion. The smallest tested inference-complete closure rather than everything
the package's metadata declares — upstream also lists a server and a CLI that
the import path Voice Chat uses does not need. `torchao` is deliberately absent
until Gate P-5 has measured INT8 on release hardware, because adding an optional
accelerator before then would be pinning a dependency nothing has run.

---

## The gates, and the tables nobody has filled in

The architecture, the installation transaction, the voice lifecycle, the
streaming path, the delivery DSP, the drain and the three-engine core are
implemented and tested. What is outstanding is *measurement*, and none of it can
be made on a machine that has none of this installed. The tables below are empty
on purpose: a number invented here would be a number somebody quoted later.

**Gate P-0 — native backend.** A throwaway harness outside product integration
proving, on the target Windows CPU: local-path-only model load; official state
load; local reference conditioning; state export and reload; first PCM before a
unit completes; drain-unit interruption reaching worker-ready with no overlap;
repeated sequential turns on one model; clean exit while generation is active;
and a cached base state still reusable across generations.

**Gate P-1 — runtime closure.** The exact Windows/Python closure installing from
empty staging without pip, and passing its self-test. No unreviewed wheel
fetched dynamically.

**Gate P-2 — streaming shape.** First PCM materially before full-unit synthesis
completes, or the runtime is not logged as streaming. `I-PKT-9` is explicit that
this is measured behaviour and not inferred from a method name.

**Gate P-2B — unit shape and internal backpressure.** One safe unit per call
against bounded coalescing against a whole bounded paragraph, comparing first
PCM, inter-unit gaps, RTF, peak RAM, stop-to-ready, and CPU activity after the
outer playback is muted. V1 keeps one unit per call unless this proves better
*without* creating long Stop waits.

**Gate P-3 — drain interruption.** Playback silent immediately; only the current
native unit finishing; its output discarded; no later unit entering Pocket; no
overlap; `stop_to_ready_ms` and the interrupted unit's size recorded.

**Gate P-VOICE-1 — precision compatibility.** A state prepared at full precision
loaded under INT8 and the reverse, checking dimensions, validation and identity.
Until it passes, precision stays in the state fingerprint.

**Gate P-11 — cached voice-state immutability.** Multiple independent utterances
from one cached state, each beginning from the same reusable base.

### Configuration envelope

| Configuration | Cold load | Warm TTFA | RTF | RAM | Stop→ready |
| --- | --- | --- | --- | --- | --- |
| Pocket full / 1 step | | | | | |
| Pocket INT8 / 1 step | | | | | |
| Pocket full / 2 steps | | | | | |
| Pocket INT8 / 2 steps | | | | | |
| Kokoro released policy | | | | | |
| Sopro released policy | | | | | |

### Unit shape

| Shape | First PCM | Gap p95 | Peak RAM | Stop→ready |
| --- | --- | --- | --- | --- |
| 1 safe unit / native call | | | | |
| first immediate + coalesced | | | | |
| bounded paragraph | | | | |

### What a Stop costs

| Interrupted unit | Chars | Audio ms | Synth ms | Stop→ready ms |
| --- | --- | --- | --- | --- |
| short opening | | | | |
| ordinary sentence | | | | |
| long allowed unit | | | | |

### Clone preparation

| Reference | Prepare ms | State bytes | Preview TTFA | Similarity notes |
| --- | --- | --- | --- | --- |
| 5 s clean | | | | |
| 10 s clean | | | | |
| 15 s clean | | | | |
| 10 s phone | | | | |
| 10 s cleaned phone | | | | |

Do not average cold and warm turns together. Do not auto-select a setting or a
batching policy from a single unexplained number: measurements inform shipped
choices, they do not silently mutate them.

---

## What is deliberately not built

* No local copy of upstream's unmerged cancellation. `interrupt_mode` stays
  `drain_unit` until a merged upstream change is reviewed and adopted.
* No GPU. The worker's environment empties every graphics variable before Torch
  is imported and the handshake refuses a non-CPU provider. That is a support
  decision for the first integration, not a claim about what upstream can do.
* No thread slider, no automatic tuner, no automatic precision or step
  selection, and no Speed cap derived from a measured real-time factor.
* No per-character model, precision or step count.
* No Voice Lab, no starter voices, and no invented emotion or style axes.
  Pocket's noise clamp and EOS threshold are not style controls, and a slider
  claiming otherwise would be a product promise nobody has tested.
* No network voice references at speech time, and no third-party mirror as a
  default trust root.
* No sherpa-onnx Pocket. It stays useful as a benchmark comparison and as a
  possible fallback if native packaging proves impossible, and it is not part of
  V1 unless a separate release decision selects it.
* No sharing of one PyTorch process between Sopro and Pocket, and no sharing of
  one worker between any two engines. Different closures and different native
  libraries are a reason for separate processes, not a code smell to unify.
