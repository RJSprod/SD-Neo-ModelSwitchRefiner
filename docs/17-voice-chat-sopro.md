# Voice Chat: a second text-to-speech engine

Voice Chat had one text-to-speech engine, and everything above the worker was
allowed to know that. "Voice" meant a Kokoro registry id, "the runtime" meant
the sherpa-onnx sidecar, and the delivery sliders were named after what Kokoro
could be asked for. This change ends all three assumptions and adds Sopro V2
Turbo behind them: a streaming, reference-conditioned model that makes a voice
from a short recording, installs itself, and runs on the CPU out of a runtime
of its own.

The user-facing rule is one sentence, and everything below is the machinery
that makes it survivable:

> **One text-to-speech engine is selected for the whole WebUI at a time.**

When Kokoro is selected, Kokoro's settings are the text-to-speech settings
everywhere. When Sopro is selected, Sopro's are. The other engine appears as a
name in the selector and nowhere else — not collapsed, not hidden behind CSS,
*absent from the document and from the payload that built it*.

Speech-to-text is not in that selector and never will be. Whisper has its own
model, its own process and its own quality tier, and switching between Kokoro
and Sopro does not reload it, change it or touch the microphone.

---

## What is new, in the order somebody meets it

**Settings → Voice Chat** gains a *Text-to-speech engine* row with two cards.
Choosing one cancels any speech, stops whichever worker was running, saves the
choice and reloads the page — so the document that comes back never contained
the other engine's controls.

**Selecting Sopro before installing it is allowed.** That is the state in which
Sopro's own page can explain itself and offer to install. Kokoro does not
reappear as an operational panel because Sopro is not ready yet, and nothing
switches back on its own.

**Install Sopro** fetches a pinned closure — Sopro 2.0.5, PyTorch 2.11.0 CPU,
torchaudio, NumPy, safetensors, SentencePiece, soundfile and their
dependencies, eighteen wheels and about 141 MB — unpacks them into an isolated
interpreter, runs a self-test, and only then promotes it. Then it fetches the
model artifacts. Both halves have a manual "install from a folder you filled
yourself" path beside them, and both check what they are given against the same
hashes the download is checked against.

**Clone voice** takes five to twenty seconds of one clear speaker, from the
browser microphone or a WAV file. The recording is validated, normalised to 24
kHz mono, and handed to the Sopro worker, which prepares the canonical
conditioning, writes it, reads it back *from the files it just wrote*, and
streams an audition through the production path. Only then is the voice
registered. The audition you hear is that one, not a fresh take.

**The Sopro voice list** does what Kokoro's does — audition, set as default,
rename, assign to a character, delete — plus *Rebuild*, for a voice whose
preparation no longer matches the installed build.

**Voice Lab (experimental)**, a closed section at the bottom of Sopro's page,
is where the eight `style_ctrl` latents and Conditioning Blend can be
investigated. Nothing in it can be saved.

---

## The architecture, and why each boundary is where it is

### Voice identity

Every voice id now says which backend owns it:

```
kokoro:official:af_heart
kokoro:clone:<uuid>
sopro:clone:<uuid>
```

Backend first, always, so no caller outside an adapter can be handed a voice
and not know whose it is. `tts_begin` carries this id on **both** engines;
Kokoro's numeric sherpa speaker rides beside it as an adapter-private field and
exists nowhere else. `mc_voice_turn` holds an opaque handle it never looks
inside, which is what makes "no shared caller depends on a Kokoro SID" a
property of the code rather than a rule somebody follows.

Legacy ids — `official:af_heart`, `clone:<uuid>`, and the bare speaker names V1
wrote — are Kokoro's, read as Kokoro's, and are **not** rewritten on sight. A
character file written in 2025 resolves correctly today and is only rewritten
when somebody edits it.

### Per-engine state

```
model_chain_tts_engine              kokoro | sopro

model_chain_voice_tts_voice_id      Kokoro default voice
model_chain_voice_speed/pitch/…     Kokoro delivery

model_chain_voice_sopro_voice_id    Sopro default voice
model_chain_voice_sopro_speed/…     Sopro delivery and generation
model_chain_voice_sopro_precision   Sopro engine settings
```

A character stores both, flat, in its own file:

```yaml
voice: kokoro:official:af_nicole
voice_speed: 1.15
sopro_voice: sopro:clone:9f2c…
sopro_speed: 0.95
```

Absence is the ordinary state and is not a missing value: a character with no
Sopro entry follows Sopro's current defaults. Editing a character while one
engine is selected copies the other engine's fields through unread, so saving
Ada's name cannot silently clear the Sopro voice she was given last week.

### Two closures, two processes, two lifecycles

```
Forge / Voice Chat parent
    ├── STT worker            Whisper, sherpa-onnx, independent of all of this
    └── active TTS worker     Kokoro / sherpa-onnx
                          OR  Sopro / PyTorch
```

`sopro_worker/worker.py` is a separate file from `voice_worker/worker.py` on
purpose. They are launched by *different interpreters out of different
closures*, and one file that had to be importable under both is one import away
from a Torch runtime reaching for sherpa. They share the wire format by
agreement rather than by import, and `tests/test_voice_sopro_worker.py` holds
that agreement to byte equality.

### The five doors, arranged twice

Both workers are stopped by extension unload, `atexit`, chained SIGINT/SIGTERM,
pipe EOF, and an OS parent-death mechanism — a Windows job object with
`KILL_ON_JOB_CLOSE`, or `PR_SET_PDEATHSIG` inside the child on Linux. A worker
whose handshake does not confirm the containment its platform requires is
stopped and the start fails.

`tests/test_voice_sopro_shutdown.py` starts real parents and real workers and
ends them in each way, including a `SIGKILL` delivered while the worker is
inside a busy loop that is not reading its pipe. That last one is the only test
that exercises the OS mechanism, and it is written so that it *fails* when the
mechanism is removed — which it did not, at first, because the fake worker was
not arranging containment and the kill was landing while it was still blocked on
a read. Both of those are fixed, and the gate was re-run with the real worker
sabotaged to confirm it now fails.

### CPU only

Sopro is created with `device="cpu"`, refuses to load anywhere else, and runs
with `CUDA_VISIBLE_DEVICES` and its siblings emptied so a Torch build that would
have found a GPU finds no devices to enumerate. The installer's self-test
refuses an installation whose Torch reports `cuda.is_available()`.

The thread policy is four intra-op and one inter-op, fixed in two constants,
reported in the handshake and in every benchmarkable log line, and never tuned
from measurements. Four is the same budget Kokoro's synthesis lane has, which is
what makes a same-machine comparison mean anything.

---

## Speed, and the thing it is not

Sopro V2 has no speaking-rate parameter. `stream()` takes `lang`,
`temperature`, `top_p`, `top_k`, `steps`, `max_seconds`, `min_seconds` and
`chunk_frames`, and none of them is speed.

So Sopro's Speed is Voice Chat's own arithmetic on the model's output, and the
obvious implementation — resample and call it speed — is the one that must not
ship, because it transposes the voice. What is implemented instead is streaming
SOLA: the signal is cut into 20 ms windows, each is shifted to the position that
best correlates with what has already been written, and the overlap is
cross-faded with a periodic Hann. Moving the analysis hop while holding the
synthesis hop fixed changes the duration; the correlation search keeps the pitch
periods aligned.

Pitch is an independent linear resample composed *after* it, so:

```
time-scale asked of SOLA  =  speed / pitch_ratio
resample ratio afterwards =  pitch_ratio
```

Measured on a 220 Hz tone pushed through in awkward 431-sample chunks:

| Speed | Pitch | Output | Scale | Fundamental |
|------:|------:|-------:|------:|------------:|
| 1.00 | 0 st | 2.000 s | 1.000 | 220.0 Hz |
| 1.25 | 0 st | 1.610 s | 1.242 | 219.9 Hz |
| 0.80 | 0 st | 2.510 s | 0.797 | 219.9 Hz |
| 1.00 | +2 st | 2.013 s | 0.993 | 246.8 Hz (want 246.9) |
| 1.25 | +2 st | 1.613 s | 1.240 | 246.8 Hz |
| 0.80 | −5 st | 2.523 s | 0.793 | 164.9 Hz (want 164.8) |
| 1.50 | +7 st | 1.342 s | 1.491 | 329.5 Hz (want 329.6) |

Speed does not move the fundamental. Pitch does not move the duration. The
largest sample-to-sample step in the output is the same as the source's own, so
there is no click where one streamed chunk meets the next — the buffer, the
fractional read position and the overlap tail all survive between calls.

At speed 1.0 and pitch 0 neither object is constructed at all.

### Speed costs real-time factor, one for one

The consequence of Speed being DSP rather than a model parameter, and it is not
obvious from the slider: the model still generates the full-length audio. Speed
throws part of it away afterwards. So the compute is unchanged and the result is
shorter, which multiplies the real-time factor by exactly the speed.

That is the difference between an engine that streams and one that stutters.

### The cost model, fitted

Synthesis time here is very nearly affine in the audio produced. Fitted over 39
segments from real conversations on one Windows machine (Torch 2.11 CPU, four
intra-op threads, full precision), spanning 0.9 s to 20 s of audio and three
different Speed settings:

```
synth_ms  =  420  +  0.798 x model_audio_ms          R² = 0.990
```

Two numbers, and they are moved by different things:

| | what it is | what moves it |
|---|---|---|
| **420 ms** | fixed cost of a unit — prompt state, first chunk | *longer segments*, not threads |
| **0.798** | marginal cost of one more second of speech | threads, precision, solver steps |

For a typical 7-second segment that is RTF **0.858** at Speed 1.00 — and since
Speed multiplies it, the **break-even Speed is 1.17x**. Below that the producer
gains headroom every second and the prebuffer is never touched again. Above it
the producer loses ground for as long as the reply lasts: at Speed 1.35, 6.1
seconds of silence owed across a 38-second reply. Invisible on short replies,
unmissable on long ones — which is exactly how it was reported.

### The first swept machine

Eight configurations, Windows, Torch 2.11 CPU, one 16-thread desktop:

| Precision | 2 threads | 4 threads | 6 threads | 8 threads |
|---|---:|---:|---:|---:|
| **full** | 0.900 | 0.824 | 0.803 | **0.705** |
| **int8** | 1.426 | 1.073 | 1.009 | 0.915 |

(RTF at a 7-second segment. Break-even Speed is its reciprocal: full at 8
threads streams cleanly up to **1.42x**, int8 at 2 threads only to 0.70x.)

Three things fall out of it.

**INT8 is slower at every thread count** — by 28% at the matched released
policy. The control used to be labelled "INT8 (faster, CPU only)"; that claim
was never measured, and it is wrong here in the other direction. Quantization
shrinks the weights; whether it shrinks the *time* depends on whether this Torch
build has int8 kernels for these shapes on this CPU, and on a small
autoregressive model the dequantize-requantize traffic can cost more than the
narrower multiply saves. Neither option claims a speed now.

**Thread scaling is poor and not monotone in shape.** 2→4 buys 8%, 4→6 buys
2.6%, and 6→8 buys 14.6%. A compute-bound workload does not look like that; this
one is bound by something else for most of its range, and the released four
threads sits exactly in the flat part.

**Four was a defensible choice for the wrong reason.** It was picked to match
Kokoro's synthesis lane so cross-engine comparison meant something, and that is
still worth having — but on this machine it leaves about 15% on the table.

Quantization shrinks the weights. Whether it shrinks the *time* depends on
whether this Torch build has int8 kernels for these shapes on this CPU, and on a
small autoregressive model the dequantize-requantize traffic can cost more than
the narrower multiply saves. So neither option claims a speed any more: full is
described as the one to compare against, INT8 as the smaller one, and the turn
summary reports the real-time factor whichever is selected.

### Measuring it, which I-12 asks for and nobody had done

I-12 wants a policy that is **measured**, fixed, and never auto-tuned from
runtime measurements. Only the third was actually true — four intra-op threads
was chosen to match Kokoro's synthesis lane so that a same-machine comparison
between the engines meant something, and never measured against six or eight.

**Run validation**, in Voice Chat Settings → Engine settings, is the missing
measurement. It spawns the isolated interpreter once per configuration — a fresh
process every time, because OpenMP sizes its pool at the first parallel region
and `set_num_interop_threads` refuses outright after one, so a sweep that reused
a process would be measuring the first thread count several times under
different labels — speaks a short line and a long one at each, fits the model
above, and writes the break-even Speed for every configuration to
`model_chain.log`.

It began as `tools/sweep_sopro_threads.py` and that was the wrong shape. A
script needs Forge's own interpreter, run from the Forge root, with Forge's data
root resolvable; all three are invisible until one is wrong, and when one *is*
wrong it reports "the isolated Sopro runtime is not installed" about a runtime
that is installed perfectly well — which is what happened the first time anybody
tried to run it. Nothing about the measurement wanted to be a command line. The
button runs in the process that already resolved those paths, so it cannot
resolve them differently.

It changes nothing itself, and the released `INTRAOP_THREADS` still only moves
by a deliberate edit. What acts on the table is a **CPU threads** control beside
Precision, defaulting to the released four.

That control is where this belongs, and the environment variable it replaces was
the wrong shape for it. Precision is already a user setting that changes compute,
RAM and which warmed caches survive; thread count is the same kind of thing, and
there was never a principle making one a dropdown and the other an environment
variable somebody has to set on Windows. I-12 forbids the *code* picking a number
from a measured real-time factor — a person reading a table and choosing a row is
precisely the "one measured, fixed policy" it asks for. The list stops at the core
count the machine reports, because offering more threads than cores is offering a
slower row with a faster-looking number, and the released value is always in it so
the shipped configuration stays reachable.

`MC_SOPRO_INTRAOP_THREADS` remains, now as the channel the parent uses to hand
the chosen count to the child — so the pool OpenMP sizes and the count Torch is
given are one number by construction rather than by two functions agreeing. A
worker running anything but the released policy says so in a warning line, in the
handshake's `thread_policy` field, and in every log line that already carried a
thread count.

### A confidence figure that could not fail

The first table this printed reported R² = 1.000 on all eight rows. It was not a
good fit; it was two observations and two parameters. A line through two points
fits them perfectly whatever they are, so the column could not have printed
anything else — a reassurance sitting next to numbers somebody was about to
change a released constant with.

The fit now returns no R² at all below three observations, and the default sweep
does two runs per length so there is something left over to check. The rule is
worth stating generally: a goodness-of-fit number computed from an exactly
determined system is not weak evidence, it is not evidence, and printing it is
worse than printing nothing.

Two things follow, and both are implemented rather than written down:

* The turn summary says so. When RTF exceeds 1, `_log_shortfall` writes a third
  line naming the seconds of silence owed, and — when Speed is above neutral —
  what the model actually produced, what came out, and what the same turn would
  have measured at 1.00x. A user whose speech stutters should not have to know
  which side of 1 is the bad side.
* The browser stops trying to hide it with a head start and starts rebuffering
  instead. See `docs/15-voice-chat-latency.md`.

What is deliberately *not* done is capping Speed, or lowering it silently when
the machine is slow. Speed above real time is a legitimate setting on a fast
machine and an unattended one on a slow one, and I-12 forbids picking a number
from a measured real-time factor. The user is told; the user decides.

---

## What is stored for a cloned voice, and what is not

```
sopro/voices/<uuid>/
    reference.wav                  the retained normalised recording
    production.safetensors         cond_vec + semantic_tokens + mel
    production.json                level_db, shapes, dtypes, schema, fingerprint
    lab-conditioning.safetensors   id_emb + style_emb + style_ctrl
```

Three files because they have three lifetimes. Conversation reads the
production asset. The Voice Lab reads the Lab asset, which may be rebuilt or
deleted without touching production. The WAV outlives both and is what either
can be rebuilt from.

**`PromptState`, per-layer K/V buffers and `StreamSession` are never written.**
They are warmed worker cache — tens of megabytes at the reviewed topology,
valid only for the solver steps and chunk size they were built with — held in
bounded LRUs of four references, three prompt states and two Lab entries. A
worker exit discards all of it.

safetensors rather than `torch.save`, because a saved voice is a file a user
keeps, syncs and restores, and `torch.save` is a pickle — which would make
"restore an old voice" a code-execution decision. Every tensor is checked
against the name, shape and dtype the metadata promised, and against a size
ceiling, before a byte of it is allocated.

`prepare_reference()` throws away `id_emb`, `style_emb` and `style_ctrl` — the
public `Reference` keeps only the projected `cond_vec` — so the adapter
reproduces the pinned preparation path to retain them. That coupling is
covered by the preparation fingerprint, which is derived from the wheel closure
hash, the installed Sopro and Torch versions, the verified model artifact
digests and the adapter's schema version. Precision is deliberately *not* in it:
INT8 quantizes the autoregressive blocks, and the encoders that produce a
voice's conditioning are untouched, so a voice prepared at one precision is
correct at the other. What precision invalidates is the warm cache, which is
keyed separately.

A voice whose fingerprint no longer matches is not guessed compatible. It is
absent from the worker's catalogue, shown as needing a rebuild, and rebuilt
transactionally — new assets written to a staging directory, validated by a
production audition, and only then does the metadata switch.

---

## The Voice Lab, and why it cannot become a setting

Sopro's speaker encoder produces an 8-dimension `style_ctrl` vector. Those are
learned latents. Nobody has measured what they mean, and the Lab exists so that
somebody can — without the investigation becoming a product claim by accident.

The isolation is structural rather than promised:

* a Lab session is a `Session` object, which no production function accepts;
* the sliders are bounded *deltas* on the saved vector, never a replacement, so
  Reset All is exactly "every offset to zero";
* auditions go through a different worker operation that returns a WAV and
  never a turn;
* the tensors it touches are `clone()`s, so nothing it does can reach the
  cached production reference;
* it writes nothing — no option, no character, no asset;
* sessions live in memory and are dropped on reload, on engine switch and on
  exit.

There is no *Apply*, no *Promote*, and no route one could call if there were —
`tests/test_voice_lab.py` asserts the module has grown no such function.
Promoting a proven conditioning method is a later design change with named
semantics, bounds, migration and tests.

Conditioning Blend recombines the three pre-projection speaker components while
holding the first voice's semantic tokens and reference mel. That is not proven
identity or style disentanglement, which is exactly why it is called
Conditioning Blend and why the surface says so.

---

## Failure behaviour

The selected engine failing is never permission to use the other one.

| Situation | What happens |
|---|---|
| Sopro selected, not installed | Written conversation works. Auto Speak is quiet. Sopro's page offers the install. Kokoro's settings stay hidden. |
| Sopro selected, worker will not start | Written conversation works. Status says Sopro could not start. No Kokoro fallback. |
| Character's Sopro voice deleted | Resolves to the Sopro default, warns in the editor. Never to Kokoro. |
| No Sopro voice exists at all | Voice says one must be created. |
| Engine switched mid-reply | Speech cancelled, old worker stopped, choice saved, surfaces redrawn. The reply does not finish in a different voice. |
| Engine switched while a reply waits to be spoken | The pending reply is refused with a sentence rather than spoken by whichever engine is selected now. |
| A page from before a switch mutates something | Refused with an active-engine mismatch; the browser fetches the surface again and replaces its panel. |

---

## What the second real attempt disproved

The install worked. `engines.installed("sopro")` returned true on the user's
machine — the traceback in their log is from *past* that check — so the runtime
and the model were both there. What they could not do was create a voice, and
the log said nothing about why.

### The reference decoder refused the file they had

`normalize_reference` accepted `WAVE_FORMAT_PCM` at exactly sixteen bits and
nothing else. The recording was a `.wav` that had been through an editor, and
that is the one case where a WAV is almost never plain: editors write 24-bit or
32-bit float as a matter of course, and a great many writers wrap even ordinary
16-bit PCM in `WAVE_FORMAT_EXTENSIBLE`, where the real format tag lives in the
first two bytes of a SubFormat GUID rather than in the format field. All three
were refused.

That was the wrong boundary. Sopro wants mono 24 kHz PCM16 and this function's
whole job is to produce it — it already downmixes stereo and resamples any rate
between 8 and 192 kHz. Narrowing a sample is the same kind of work. It now
decodes 8-, 16-, 24- and 32-bit PCM and 32- and 64-bit float, unwraps
`EXTENSIBLE`, clamps float samples that exceed unity rather than letting them
wrap, and treats a NaN as silence. What is still refused is a *compressed*
encoding, which needs a codec this module should not grow — and the refusal now
names what the file is (`IMA ADPCM`, `encoding 0x2001`) rather than only what
was wanted, which is the half of the sentence somebody can act on.

### None of it was written down

The clone route answered the browser with an exact reason and logged nothing.
The install path logs every step; the clone path logged only success. So the
log that arrived said a reply had not been spoken and nothing about the three
attempts to create the voice that would have spoken it. Refusals are logged
now, with the attempt in front of them.

Two smaller things in the same report, both reporting rather than behaviour:

* The start-up summary described the sherpa runtime, Whisper and Kokoro
  whatever engine was selected. It now also says which engine *is* selected
  and, for Sopro, whether its runtime and model are installed and how many
  voices exist — so the first line of a log answers the first question anybody
  asks of it.
* "No voice has been created yet" reached the log as a full traceback at
  WARNING, on every assistant turn, for as long as the state lasted. It is a
  state the design's own failure table calls ordinary; it is now one throttled
  sentence, with the traceback kept at debug.

---

## One control per value

A host option is two things at once: a stored value, and a component on the
settings page. Forge's "Apply settings" writes every component on that page back
into the store, and the browser's copy of each was stamped when the page was
built — so it knows nothing about anything a live panel changed since.

Every one of Sopro's twelve settings was therefore duplicated: once in the panel
meant to be used, once as a row further down the same page, each able to
overwrite the other. Setting a default voice and then changing anything else on
that page put the old default quietly back, and Conversation went on speaking
with the first voice in the list. The same was true of the delivery sliders and
the engine settings.

They live in Sopro's own files now — the default voice beside the voices in
`registry.json`, the rest in `settings.json` — written atomically, out of reach
of any settings form, and set in exactly one place. The options are still *read*
once, so an installation configured under an older build keeps what it chose.
Kokoro's default voice had the identical defect and moved to its own registry
file with it.

## Two controls that did not remember

**The character's sampling.** Temperature, top-p, reply tokens and seed already
*loaded* with a character. Nothing wrote them back without pressing Save
character — which is on the editor screen, while the accordion holding them is
in the flyout, beside whoever you are talking to. So setting a seed of −1 there
set it for that reply and for nothing afterwards. They are written back on
change now, while the editor is shut; with it open they belong to the edit, and
Save and Cancel decide their fate.

**Where the page was.** Emptying a list and refilling it costs the scroll
position of whatever is scrolling it: the browser clamps `scrollTop` to the
content present at that moment, a list wiped to nothing clamps to zero, and
refilling it does not put it back. Every poll that repainted a voice list threw
somebody back to the top of the flyout, and expanding a section — which is
followed by a repaint — did it reliably enough to look like the expanding caused
it. Every rebuild now runs inside a helper that notes where the page and the
nearest scrolling ancestor were and puts them back.

## Cleaning a reference recording, and the denoiser that could not be installed

Sopro clones what it is given, hiss included. DeepFilterNet was the obvious
answer and **cannot be installed here**: its Rust extension, `DeepFilterLib`,
publishes wheels for CPython 3.8 to 3.11 only, and this WebUI runs 3.13.
Building it from the sdist needs a Rust toolchain — precisely the "resolve and
build something nobody reviewed" that the pinned-wheel design exists to prevent.
That is a fact about the package, not a preference, and it would apply to any
isolated engine process built around it.

What is there instead is spectral subtraction, in the page, on the selection
about to be uploaded: an 80 Hz high-pass, a per-bin noise floor estimated as the
10th percentile over time (minimum statistics, because a recording somebody
trimmed themselves rarely opens on silence), moderate over-subtraction, a gain
smoothed across frequency and time, and a peak normalise.

The smoothing is the part that matters. An unsmoothed gain flips between "keep"
and "floor" bin by bin and frame by frame, and the result is musical noise — a
shimmer of tones where there used to be honest hiss. Smoothing is what buys the
moderate over-subtraction; without it the only way to remove this much noise is
to be aggressive enough to hear. Measured on a synthetic speech-like signal it
is about 6 dB, and `tests/test_voice_chat_js.py` asserts both halves of that:
the hiss falls, and the voice is still the loudest thing in the result.

It is not a learned denoiser and the UI does not claim it is. It takes out what
is steady and leaves what is not, which is most of what a bad reference
recording suffers from, and it costs no dependency at all. One thing it is
correctly bad at is a genuinely constant tone — which is indistinguishable from
stationary noise, and which the first version of its test used as the signal.

### Three things a denoiser cannot fix

None of these is noise, so none of them is touched by any amount of spectral
subtraction, and all three are ordinary in a recording made on a phone.

* **Clicks.** One sample where the waveform was never going. Detected as a
  second difference far larger than the first differences either side of it —
  band-limited sound takes large steps but changes those steps slowly, and a
  plain amplitude threshold would take the top off every plosive instead.
* **Clipping.** A run of samples pinned at the rail is a peak the recorder could
  not write down; left alone it is a square wave in the reference, which is
  broadband, harsh, and faithfully learned by the voice built from it. An arc
  goes back over it, bulging in proportion to how long the flat top ran.
* **Level.** Peak normalisation alone turns the speech down to make room for one
  door slam. The target is an RMS level with the peak ceiling only there to stop
  it clipping on the way out.

The levelling test found a real bug in the stage above it: the first and last
window's worth of samples have only partial overlap-add weight, so dividing by
that weight amplified them into a spike at each end — which was then the loudest
thing in the clip and dragged the whole gain down. The divisor is floored at half
the steady-state weight now, and the ends are faded.

### Why not a learned denoiser, yet

DeepFilterNet is the right destination and the objection first recorded here —
that it "cannot be installed" — was too absolute. `DeepFilterLib` ships
`win_amd64` wheels for CPython 3.10 and 3.11, and this extension already builds
isolated environments; nothing stops one of them being built from a different
interpreter. What that costs is a second CPython *and* a second copy of Torch,
since DeepFilterNet's inference path imports Torch and cannot share the cp313
one Sopro already has: roughly 150 MB and a second runtime, for cleaning a
twenty-second clip.

Two things stop it being built here rather than being merely expensive. A
redistributable interpreter cannot be pinned from this workspace — only PyPI is
reachable, and every other byte this feature installs is checked against a hash
committed in the repository. And none of it could be executed even once: no
Windows, no cp311 Torch, and the weights are behind the same blocked host.

RNNoise was measured rather than assumed, and the measurement is the reason it
is not here either. On a synthetic vowel plus hiss its own speech probability
stayed under 0.5 for 98% of frames: it decided the signal was not speech and
gated it away, leaving 3% of the voice band and a correlation with the clean
reference of −0.003. That is very likely the synthetic signal's fault rather
than a fair verdict on real speech — but it is exactly the failure mode a
VAD-gated model has, and a reference recording is arbitrary material somebody
brings from outside. On the same signal the spectral pass moved correlation with
the clean voice from 0.868 to 0.967 and took about 70% of the hiss out, because
it has no opinion about what speech is.

So the order was: a signal-agnostic pass that cannot destroy a recording, first;
a learned one after it. The second half is now built.

### DeepFilterNet, on an interpreter of its own

The owner chose to have it, and chose how the interpreter should be trusted, so
here it is: a third isolated runtime, installed and removed on its own, started
when a recording is being cleaned and stopped two minutes after the last one.

It is **not** a text-to-speech engine and is deliberately outside the selector
(I-1). Cleaning a recording is not speaking, the choice of speaker has no
bearing on it, and a third row in a selector that says "one engine speaks at a
time" would be a third thing that could be "the engine". `mc_voice_cleanup` is
reachable from either engine's clone form and from neither engine's state.

**What it costs, exactly.** 23 pinned wheels totalling 228 MB, of which
`torch-2.2.2-cp311-win_amd64` is 198 MB, plus a ~10 MB interpreter and a ~3 MB
model: **242 MB**. The second Torch is unavoidable — `DeepFilterLib` is a Rust
extension published for CPython 3.8 to 3.11, so this cannot share Sopro's cp313
one — and the number is on the settings row rather than behind it, because a
quarter of a gigabyte to tidy a twenty-second clip is a decision somebody should
make before pressing anything.

**Why torchaudio 2.2.2 and not the newest.** `df/io.py` imports
`torchaudio.backend.common.AudioMetaData`, which torchaudio removed after 2.2.
Pinning the newest pair would install cleanly and fail on the first import. A
shim that satisfied the import was considered and rejected: it would have saved
90 MB by allowing torch 2.11, and nothing here can be executed before it reaches
the owner's machine, so the version DeepFilterNet was actually written against
is worth more than the saving.

**The interpreter is the one unpinned executable.** python.org is not reachable
from the workspace the manifest is generated in, so its digest is recorded on
the machine that first fetches it rather than checked against a constant
somebody reviewed — the same path the model artifacts already take, agreed to
explicitly, and weaker than the 23 wheels around it. It is said in the module
docstring, in the manifest, and in a test.

**What could not be verified here**, and it is the whole of the risk: there is
no Windows, no cp311 Torch and no model in this workspace, so nothing below the
model load has ever run. What the tests assert is everything up to it — the
manifest is complete and pinned, the platform gate is closed, the install
transaction refuses before touching anything, the framing round-trips against
the *Sopro* worker's reader, the handshake refuses a wrong backend, a wrong
protocol, a GPU device and a Linux worker that cannot confirm containment, and
shutdown is safe when nothing is running. The first real execution is the
install button.

---

## Containment, and the check that was in the wrong place

The install worked, the decoder took the file, and then the worker would not
start: *the Sopro worker could not be tied to this process's lifetime.* Three
times, across a restart.

The parent creates a job object with `JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE` and
assigns the worker to it. Both calls returned success — that is the arrangement,
and it is what the kernel enforces. The refusal came from somewhere else: the
worker checked its own containment with `IsProcessInJob` through the
`GetCurrentProcess` pseudo-handle, and reported the same word, `pipe`, whether
the kernel said "no" or the call could not be made at all. The parent read that
as "no" and refused. A worker whose containment was arranged and *was being
enforced* could never start.

Two things were wrong and both are fixed.

The check is now made by the parent, immediately after the assignment, with
`IsProcessInJob(child, job)` — real handles on both sides, and it names *this*
job rather than asking the weaker "any job at all". That is stronger evidence
than the worker could ever produce, and it is available at the moment and place
the arrangement is made. Windows containment is proved there; the worker's own
answer is logged and is not a veto. On Linux it is the other way round —
`PR_SET_PDEATHSIG` can only be set inside the child — so there the child's word
is still the only evidence there is and still has to be good.

The worker also now distinguishes `job`, `none` and `unknown` instead of
collapsing the last two, and every Win32 call on both sides declares its
`argtypes`: a HANDLE is not a C `int` on 64-bit Windows, and the calls here
survived that only because kernel handles are small — which the pseudo-handle
`-1` conspicuously is not.

The worker's own diagnostics were being logged at `debug`, so the sentence
explaining which of the three had happened was invisible at the default level in
the log of the person it stopped. They are `info` now.

---

## Bringing audio in, and getting started without a microphone

Two walls in front of the first voice, both of them removed in the page rather
than in the model.

**Any format, trimmed here.** Sopro conditions on five to twenty seconds of mono
reference audio. That is a real constraint, and it was being handed to the user
as homework: find a WAV, make it 16-bit PCM, make it the right length. The
browser already decodes every format it can play, and this extension already had
a WAV encoder for dictation, so both jobs move into the tab. Choose an MP3, M4A,
FLAC, OGG or WAV — or record one — and it is decoded, drawn as a waveform, and
trimmed by dragging or with two number boxes; "Pick 15 s for me" takes the
loudest window, which beats the first fifteen seconds on a clip that opens with
silence. What is uploaded is the selection, as one mono 16-bit PCM WAV. A file
longer than the window opens on a selection that will work rather than on a
refusal. Nothing about this reaches the network, and the server's validation is
unchanged and still strict.

**Starter voices.** Sopro has no speaker bank — every voice it has comes from a
reference recording — so a fresh installation offered exactly one way forward:
record yourself. That is a wall in front of somebody who only wants to hear
whether the engine works. There is now a button that makes four voices by having
Kokoro read a short passage and cloning that.

This is the one place the two engines touch, and it is a *creation* step rather
than a dependency: what it leaves behind is an ordinary Sopro voice with its own
retained reference, which renames, rebuilds, auditions and deletes like any
other and keeps working if Kokoro is removed. It is also the one clone where
consent is not a question anybody has to weigh — a Kokoro speaker is synthetic
and Apache-2.0, so no person's identity is being copied. Section 8's rule about
bundled example voices is untouched: nothing ships with the extension,
`sopro.official()` is still empty, and a starter voice is a local clone that
says so in its name.

**Contrast.** Every field this feature adds now states its own contrast pair and
means it. The previous rule set `background-color`, which a host theme's
`background` shorthand beat on specificity — so LobeTheme's night mode went on
painting a white box under light text. Both halves now come from one place: the
text is whatever is already legible on the surface and the field is that same
colour at 8%, which computes to about 11:1 on a dark theme and 15:1 on a light
one without this stylesheet knowing which it is in. Borders clear 3:1,
placeholders 5:1, focus is always visible, and the file button, checkboxes and
sliders are covered too.

---

## Two things the first real install disproved

Both were found by installing this on Windows rather than by reading it, and
both were assumptions the design stated confidently and got wrong. They are
recorded here because in each case the *mechanism* was sound and the *fact it
rested on* was not, which is the kind of mistake that comes back.

### An entity tag is not a checksum

An unpinned model artifact is resolved against the publisher before it is
fetched, and what was read as the publisher's digest was the `ETag` of whatever
answered the HEAD. Two things were wrong with that.

The HEAD followed redirects. Hugging Face answers `/resolve/` with its own
headers — `X-Linked-Size`, and `X-Linked-Etag`, which *is* the LFS object's
SHA-256 — and a 302 to a delivery host. Following it discarded the publisher's
answer and read the delivery host's headers instead. `huggingface_hub` follows
only *relative* redirects for exactly this reason.

And the fallback accepted a bare `ETag` whenever it was sixty-four hexadecimal
characters wide. RFC 9110 defines an entity tag as an opaque validator: a host
may make it an MD5, a storage-layer hash, an upload id or a random string, and
is only obliged to change it when the body changes. Hugging Face's delivery
host answers with a sixty-four character value that is not the file's digest.

Together those two produced a check that could not pass. Every artifact above
the LFS threshold — Sopro's five `.safetensors`, and Whisper's ONNX exports on
the same hub — downloaded correctly, hashed correctly, and was thrown away for
disagreeing with a cache key. Small non-LFS files were unaffected, which is why
`config.json` installed and `model.safetensors` did not.

The HEAD now stops at the first redirect that leaves the publisher's host, and
a digest is taken only from a header whose *name* says it is one
(`x-linked-etag`, `x-checksum-sha256`, `x-amz-meta-sha256`) or from a value
that says so itself (`sha256:…`). A publisher that states no digest is not a
failure and never was: the byte count is checked, the download proceeds, and
the digest of what arrived is recorded so the next install of that bundle is
checked against a constant. `tests/test_voice_models.py` runs this against two
real local hosts wired like a hub, because a fake that does not redirect cannot
tell the two readings apart — which is precisely why the original tests passed.

### A settings row cannot be refreshed by reloading

Section 5 asks that the inactive engine's operational controls be absent from
the document rather than hidden in it, and the browser honoured that by
reloading after a switch: ask the server for a document that never contained
them.

Forge does not build settings markup per request. `OptionHTML` takes a string,
and this extension's rows are built once, when the extension is imported, and
that same string is served to every page load for the life of the process. So
the reloaded page came back with markup for the engine that had been selected
at *startup*, the page compared it against the engine selected *now*, correctly
concluded it was stale — and reloaded again. The loop ended when the tab was
closed. Switching engines then required restarting the WebUI to reach the other
engine's settings at all.

There is now a `/model-chain/voice/surface` route that builds the settings and
voices markup on demand, and the browser replaces those nodes with what comes
back. That keeps section 5 exactly as written — the other engine's controls are
gone from the document because they were removed from it — without depending on
a document the host will not rebuild. Nothing in `javascript/voice_chat.js`
calls `location.reload()` any more, and a test asserts that. A cooldown bounds
the rebuild to one every three seconds, so that if the mismatch ever somehow
survives a rebuild, what is left is an occasional request rather than a browser
that will not stop.

---

## What is not done, and what a release still needs

The architecture, the installation, the voice lifecycle, the streaming path,
the delivery DSP and the Lab are implemented and tested. Three things in the
design intent are explicitly *measurements*, and they cannot be made on a
machine that has none of this installed:

* **Gate S-1's Windows correctness check** has now run. The runtime installed
  end to end on Windows 11 / Python 3.13 — eighteen wheels fetched and hash-
  checked, unpacked into an isolated interpreter without pip, self-test passed:
  *Sopro 2.0.0, Torch 2.11.0+cpu, NumPy 2.2.6, 4 intra-op / 1 inter-op threads,
  attention stable*. That is the fixed-seed SDPA repeatability check and the
  single-threaded comparison at the released thread policy, on a real Windows
  PyTorch build. What the gate still wants is the same result on more than one
  machine before it is called met. (The package reports its version as 2.0.0
  from the 2.0.5 wheel; nothing compares the two, and the preparation
  fingerprint records what is installed rather than what was asked for, which
  is the reason that is harmless rather than a coincidence.)
* **Gate S-3's latency and throughput envelope** — same-machine TTFA, RTF and
  RAM against Kokoro — needs the real closure on the real hardware class. The
  telemetry that would report it is in place and content-free.
* **Section 41's `style_ctrl` sweep.** The Lab's slider range is a conservative
  opening bound of ±1.5, documented as provisional. The useful range has to be
  measured across several voices before release, and the UI then uses the
  tested one.

The model artifact hashes are unpinned in the manifest, exactly as Kokoro's and
Whisper's are, and are resolved against the publisher at install time and
recorded — `tools/pin_sopro_models.py` turns them into constants once a
maintainer has fetched them. The bundle's declared size is now the publisher's
own total (729.2 MiB) rather than the estimate it shipped with, which was
almost twice that. The runtime closure *is* pinned, because an
unpinned wheel is a wheel nobody reviewed and the whole preparation fingerprint
rests on knowing which bytes ran.

Linux is not in the Sopro allowlist. PyPI's Linux Torch wheels pull the entire
CUDA closure through their dependency markers, and the first Sopro release
neither claims nor consumes a graphics device. A Linux CPU closure comes from
PyTorch's own `+cpu` index and is a separate, deliberate addition with its own
gate.
