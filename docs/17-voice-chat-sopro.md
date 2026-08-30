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
