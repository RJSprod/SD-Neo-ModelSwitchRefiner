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

## The click at the end of a sentence

Reported from a real machine: on a long reply, a small pop at the end of some
sentences. It is not the buffer running dry — that is counted, and it sounds
like a gap rather than a tick — and it is not the segmenter. It is the join.

PocketTTS stops a generation a couple of frames after the end-of-speech token
and leaves it there. Upstream's `_autoregressive_generation` breaks out of its
loop mid-stride, and the next generation's `_decode_audio_worker` calls
`init_states` on a fresh Mimi decoder, so:

* a unit's **last** sample is wherever the waveform happened to be, and
* the next unit's **first** sample is wherever a zero-initialised decoder puts
  it.

Voice Chat plays units back sample-exact and one after another, which turns that
pair into a step in the waveform — and a step is a click. It is small, it lands
at the end of a sentence, and on a long reply it happens once per sentence. A
residual DC offset in the decoder output, which codec decoders have, makes it
louder.

`Seam` is the answer and it is eight milliseconds long. Each committed unit is
ramped up over its first `DECLICK_MS` and down over its last, with a raised
cosine rather than a straight line so there is no corner at either end. Ramping
the *end* means knowing which block is the last one, which streaming does not
know until the generator is exhausted — so a unit's final eight milliseconds are
withheld inside `Seam` and released by `flush()` when the unit ends. Eight
milliseconds of added latency once per sentence is not audible; the click is.

What it costs is the first and last eight milliseconds of each unit, which for
this model is the padding either side of the end-of-speech token rather than
speech. What it does not do is fade per *chunk*: that would put a fade several
times a second in the middle of a word, which is why the seam is scoped to the
unit and not to the block.

The same join exists in Kokoro and in Sopro — each synthesises one unit and
stops — so each worker carries its own copy, held to its own edge tests, the
same way the speed and pitch DSP is duplicated above.

One consequence worth writing down: for Kokoro the number of blocks that leave
`Engines.stream` is no longer the number sherpa handed back, because the seam
adds a closing block of its own. The callback-granularity metric is about
sherpa's hand-backs, so it is now counted where they happen — inside
`Engines.stream` — and returned to the lane rather than inferred from how many
times the lane's `on_audio` was called.

What this does **not** fix is the other half of "it sounds like separate
chunks": each unit is its own generation, so prosody restarts at every sentence.
Upstream's own long-text path has the same seam and the same restart, and its
`TODO` about teacher forcing across chunks is the fix for it. That is a model
change, not a DSP one.

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

The manifest ships **half resolved**, and says so in its own `notes`.

The **runtime closure is pinned**: 120 wheels — thirty packages across four
Python minors — each named, sized and hashed from pypi.org, about 275 MB per
platform. So the managed runtime install fetches exactly what this repository
claims and refuses anything else.

Thirty rather than the thirteen it began as, and the difference was
**measured rather than reasoned about**. A real `pocket-tts 3.0.2` wheel — the
one whose SHA-256 this manifest pins — was installed and every candidate package
was blocked at the import hook in turn to see which ones make `import pocket_tts`
raise. The original list was written by reading upstream's imports, and it was
seventeen wheels short: five that `pocket_tts` imports at module level, one that
no environment variable can turn off, nine those bring with them, and two that
only a reading of the *declarations* finds — see below.

- `pydantic` — `pocket_tts.utils.config` builds its config model with it;
- `PyYAML` — the same module parses the config with it;
- `scipy` — `pocket_tts.data.audio_utils` imports it at module level;
- `huggingface-hub` and `requests` — `pocket_tts.utils.utils` imports both at
  module level, whether or not anything ever resolves a location;
- `beartype` — `pocket_tts/data/audio.py` does `from beartype.typing import
  Iterator` unconditionally. `POCKET_TTS_NO_BEARTYPE=1` in the worker's
  environment keeps its claw from wrapping every function in the package, which
  upstream's own comment says costs per call, but it does not remove the import.

Those six bring nine of their own: `pydantic-core`, `annotated-types` and
`typing-inspection` with pydantic; `urllib3`, `certifi`, `idna` and
`charset-normalizer` with requests; `tqdm` and `packaging` with huggingface-hub.
And `colorama` and `setuptools` complete it, for the reason the next section
gives.

### A closure where every pin is real and the set does not install

The first version of that list shipped, and it did not work. Every wheel
downloaded, every SHA-256 matched, all thirty megabytes unpacked into the
isolated interpreter, and the self-test — which runs before anything is
promoted — refused it:

```
ImportError: cannot import name 'Sentinel' from 'typing_extensions'
```

`typing-extensions` was still pinned at 4.12.2, chosen when this closure was
thirteen packages and Torch was the only thing asking for it. `pydantic` and
`typing-inspection` had since arrived needing 4.14.1 and 4.15.0. Every
individual pin was a real, verified wheel; the *set* was not installable, and
nothing in the pipeline was looking at the set.

That is now `tools/pin_pocket_models.py`'s job. It reads what every publisher
declares about every other package in the closure, evaluates the markers once
per advertised Python minor, and refuses to write a manifest whose pins
contradict one another. It still resolves nothing — that property is the whole
point of this design — it checks a written-down list against what the
publishers say, which is a different thing from asking an installer to choose.

Run against the closure as it shipped, it names all five problems at once:

```
pydantic 2.13.5 needs typing-extensions>=4.14.1 on Python 3.10, and the closure pins typing-extensions 4.12.2.
pydantic-core 2.46.5 needs typing-extensions>=4.14.1 …
typing-inspection 0.4.4 needs typing-extensions>=4.15.0 …
torch 2.6.0 needs setuptools; python_version >= "3.12" on Python 3.12, and the closure does not ship it.
tqdm 4.70.0 needs colorama; platform_system == "Windows" on Python 3.10, and the closure does not ship it.
```

The last two are the interesting ones. Reading the *import path* finds neither:
tqdm imports colorama inside a `try`, and nothing on the inference path touches
`setuptools`. Reading the *declarations* finds both — and this closure is
Windows-only and advertises 3.12 and 3.13, so both markers are live on exactly
the platforms it ships to. They are in the closure now, and the omissions that
remain are a table in the tool with a reason beside each name rather than an
absence nobody wrote down.

`huggingface-hub` is pinned in the **0.x** line on purpose: 1.x replaced
`requests` with `httpx` and would add `httpx`, `httpcore`, `anyio`, `h11` and
`sniffio` to an offline worker for an HTTP client it must never use. Having the
hub library present at all is uncomfortable and the answer is not to pretend it
is absent: `HF_HUB_OFFLINE` and `TRANSFORMERS_OFFLINE` are set in the worker's
environment, the config it is handed names only local files, and it refuses one
that names a network location anywhere in it — including three levels down,
where the tokenizer's location lives.

Torch is pinned at 2.6.0 rather than at the 3.0.2 metadata's `>=2.5.0` floor,
and the reason is the kind of thing a pinning tool exists to catch: 2.5.x has no
`cp313` Windows wheel, and this manifest advertises Python 3.13. A floor is not a
runtime identity, and a platform advertised without a wheel to satisfy it is a
platform that fails at install time instead of at review time. `scipy` is pinned
at 1.15.3 for the mirror image of the same reason: 1.16 dropped Python 3.10,
which this manifest still advertises.

The **model, official voice and cloning artifacts are declared but not hashed**,
because the machine that wrote the manifest could not reach huggingface.co. The
difference from the closure is worth stating rather than glossing: an artifact
with a digest here is checked against a number this repository committed to, and
one without is checked against the digest its publisher reports at install time.
Both refuse a file that arrives wrong; only the first refuses a publisher that
changed its mind. `tools/pin_pocket_models.py --model`, run on a machine that
can reach the hub, turns the second into the first — with `HF_TOKEN` set and the
Kyutai conditions accepted if the gated half is wanted, and a closed gate there
leaves the public half resolved and written.

Their **locations and revisions** are upstream's own statement rather than a
model card somebody read. The `english.yaml` that ships inside the 3.0.2 wheel
names three files:

```
weights_path:                        hf://kyutai/pocket-tts/languages/english/model.safetensors@39592ff2…
weights_path_without_voice_cloning:  hf://kyutai/pocket-tts-without-voice-cloning/languages/english/model.safetensors@d29db797…
flow_lm.lookup_table.tokenizer_path: hf://kyutai/pocket-tts-without-voice-cloning/languages/english/tokenizer.model@d29db797…
```

Two repositories at **two different commits**, which is upstream's arrangement
and not an oversight here. There is a **third**: `get_predefined_voice` in
`pocket_tts/utils/utils.py` resolves an official voice to

```
hf://kyutai/pocket-tts-without-voice-cloning/languages/<language>/embeddings/<name>.safetensors@e81d79e8…
```

— the same public repository as the weights, at a later commit, because the
embeddings were added to it afterwards. So the manifest carries a revision per
*artifact group* rather than per repository, and the pinner resolves each
against its own. Following `main` would install bytes that shipped
configuration was not written for.

That same file is where the official voice bank comes from. An earlier version
of this table was invented — it named `marlow`, `juno` and `rhys`, which do not
exist, under `voices/<name>.safetensors`, which is not a path the repository
serves. Upstream advertises twenty-six names in
`_ORIGINS_OF_PREDEFINED_VOICES`; which of them have an *English* embedding is a
question only the repository can answer, and the all-or-nothing rule on the
voice bank means one name that is not served costs the user every voice rather
than one. So the manifest ships the single name there is independent evidence
for — `alba`, which is also the reviewed default — and the `--model` run prints
the whole `embeddings/` directory for a maintainer to review and add. Their
accents are not in the manifest either: that is not something this repository
can source, and a made-up one is exactly what the three invented entries were.

Install-from-a-folder works either way, and what is supplied has its digests
recorded and becomes the constant the next install is checked against.

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

*Partly run, and it changed the code.* The half of this gate that needs no model
weights was executed against a real `pocket-tts 3.0.2` wheel — the one whose
SHA-256 this manifest pins — and the API this worker had been written against
was wrong in five places, each of which would have failed on the first reply
somebody wanted spoken:

| What the worker did | What released 3.0.2 has |
| --- | --- |
| `load_model(config=<dict>, device="cpu")` | no `device` parameter at all, and `config` must be a **path to a `.yaml` file** — a dict is refused with *"Config should be a path to a YAML file ending with .yaml"* |
| `generate_audio_stream(state, text, sampler_decode_steps=…, temperature=…)` | `(model_state, text_to_generate, max_tokens, frames_after_eos, copy_state)`. Both numbers are **instance attributes** read inside the sampler |
| `get_state_for_audio_prompt(<wav bytes>)` | `(audio_conditioning: Path \| str \| Tensor, truncate=False)` — bytes fall through every branch |
| `model.export_voice(...)` / `model.save_state(...)` | `pocket_tts.export_model_state(state, dest)`, a **module-level function**. `export_voice` is the name of upstream's CLI command |
| `get_state_for_audio_prompt(<str path>)` for a saved state | correct *shape*, wrong *type*: a `str` is handed to `download_if_necessary` first, which resolves `https://` and `hf://`. A `Path` goes straight to the file |

That last one is the one worth dwelling on. It was not a crash — it was the
single call in the worker that could have reached the network, in the engine
whose whole design says nothing after installation does. Passing a `Path`
closes it.

Two consequences beyond the call sites. Upstream's `Config` model **forbids
unknown keys**, so the local config can no longer be a document of this
repository's own design with a `schema` and a `model_id` in it: it is upstream's
own shipped `english.yaml` with three paths replaced, copied out of the wheel at
install time by `worker.py --recipe`, and everything else Voice Chat wants to
tell the worker goes over the wire in `worker_config()` instead. And because a
local path that is missing does not fail where upstream's fallback catches it,
`weights_path` on a machine without the gated half is pointed at the **public**
weights rather than left naming a file that is not there — otherwise "installed
without cloning" would be a model that refuses to load rather than one that
speaks with official voices.

What is still outstanding is everything that needs the weights: first PCM before
a unit completes, RTF, drain timings, and a cached base state's reusability
across generations. Those are measurements, and the tables below are still
empty.

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
