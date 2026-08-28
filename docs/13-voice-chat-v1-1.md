# Voice Chat V1.1 — implementation notes

Companion to *Model Chain / LLM Studio — Voice Chat V1.1 + Kokoro Voice
Management + Optional CPU Voice Cloning* (28 August 2026). That document states
what the behaviour has to be; this one records the choices made against it, the
two places it was deliberately narrowed, what its feasibility gates actually
established, and the handful of things that would otherwise be rediscovered the
hard way.

Section numbers below are the design intent's. `T` numbers are its test matrix,
and every one of them has a test of the same name.


## 1. What this is, in one sentence

Speech stopped being something that happens *after* a reply and became something
that happens *during* one — and everything else in the release is a consequence
of that.

Streaming is what forces the worker to be cancellable, which forces the runtime
locks apart, which is what lets Stop stop both halves, which is what makes Stop
have to stay on screen after the model has finished. The voice registry and the
bank are a second thread of work, but they meet the first one at exactly one
point: a turn resolves a stable voice id to a number once, at its creation.


## 2. §§5–11 — the segmenter, and why it is a state machine

`mc_voice_segment.py` has no I/O in it and no dependency beyond `re`. Section
116 rules out an NLP tokenizer, and the reason is the dependency rather than the
accuracy: this runs inside the WebUI process, beside an image model, and adding
a data-file-shipping package to decide where a sentence ends would be a strange
trade.

What is there instead is a boundary preference order and a named guard for each
way a full stop lies — decimals, abbreviations, initials, domains, ellipses,
open code fences — each with a test. Two decisions inside it are worth naming:

**Boundaries are found on raw text; normalization happens at emission.** The
alternative, normalizing the whole buffer on every chunk, is quadratic in the
reply length and buys nothing: `**bold**` does not create a false sentence
boundary.

**Fence parity survives a commit.** A fence opened in a segment that has already
been spoken is still open for the next one, and the pending buffer cannot know
that on its own — so the parity is carried on the segmenter rather than
recounted from what is left.

The leading-label rule (§9) is the one place the segmenter holds text back. It
holds only while a label is still *possible*: a newline, a terminator, a colon,
six words or ninety-six characters all settle the question, and after any of
them the opening is spoken as it stands.


## 3. §§12–18 — protocol 2, and the three parts of the worker

Protocol 1 had one synchronous `tts`, and its command loop was blocked inside
native ONNX code for as long as synthesis took. That single fact made two things
impossible at once: audio could not leave before the last sentence, and nothing
could be *told* to stop while it ran.

The worker is now a command loop that is always reading, one inference lane that
serializes Whisper and Kokoro, and one writer that is the only thing touching
stdout. Cancellation happens in sherpa's generation callback, which is handed
each finished sentence batch and whose return value decides whether generation
continues.

Two details that are easy to get wrong:

**Flow control is a counter, not a queue depth.** The callback waits while more
than `MAX_PENDING_AUDIO` frames are unwritten, on a condition variable that
cancellation also notifies — §16's requirement that backpressure can never make
a cancel wait behind its own producer.

**Streaming granularity is reported, not assumed.** `Engines._callback_supported`
asks whether this sherpa build's `generate` takes a callback and says so in the
handshake. With one, audio leaves during a sentence batch; without one, it
leaves at the end of each segment the parent committed. Both stream; the second
is coarser. A build that could do neither would have failed the handshake.


## 4. §§15–16 — the lock split, and what it buys

V1 held a single `RLock` around a whole request. That was correct, and it was
the reason nothing could be cancelled: a thread waiting three seconds for Kokoro
held for three seconds the lock that Stop, `status()` and unload all wanted.

`_state_lock` now covers the process handle, the handshake, the lifecycle flags
and the two registries, and is never held across a wait for anything.
`_write_lock` serializes writes to the worker's stdin. A request registers a
queue of its own under `_pending`, writes its frame, and waits on *that queue* —
which is also why two requests cannot receive one another's replies (T-RT-9)
structurally rather than by a check.

The single reader dispatches an ordinary reply to its request's queue and a
streaming frame to its turn. A frame whose turn is not in `_turns` is dropped,
which is §24's late-audio race handled at the one place every frame passes.


## 5. §§48–57 — the voice bank, and what GATE VC-0 actually established

The original cloning proposal was `Storytime .bin → torch.save .pt → register
with Kokoro`. That is how the *upstream Python* Kokoro is used. Production TTS
here is `sherpa_onnx.OfflineTts`, which does not know what a `.pt` is and has no
concept of registering a voice at all. §48 supersedes it, and `mc_voice_bank.py`
implements what replaces it.

Read out of `offline-tts-kokoro-model.cc` rather than assumed:

* the ONNX metadata carries `n_speakers` and `style_dim` (`510,1,256` for v1.0);
* `voices.bin` is loaded separately, as one flat block of float32;
* the loader refuses to start unless the block holds exactly
  `style_dim[0] × style_dim[2] × n_speakers` values;
* choosing a speaker is pointer arithmetic —
  `styles + sid × 510 × 256 + token_len × 256` — and what reaches the graph is a
  1×256 style vector;
* **a `sid` at or past `n_speakers` is not an error**: sherpa logs a warning and
  uses speaker 0.

The first four say the graph never sees a speaker, so adding voices needs a
longer `voices.bin` and a model whose metadata agrees about its length — no new
weights, no retraining, no sherpa fork, no second engine. The fifth is why every
custom voice is synthesized through the production runtime before its registry
entry is committed: a bank whose metadata had not taken would produce a clone
that works perfectly and sounds like somebody else.

### What was proven, and where

The gate has two halves and they were established in different places, which is
worth being exact about.

**The derivation half is proven here, in this repository.** `derive_model`
rewrites `n_speakers` at the protobuf level with no ONNX dependency —
`metadata_props` is field 14 of `ModelProto`, a repeated `StringStringEntryProto`
of `key` and `value`, and every other byte is copied through. It was checked
against the real `onnx` package during development: the output passes
`onnx.checker`, its initializers are byte-identical, exactly one metadata string
differs, a value whose length changes still produces a valid file, and two runs
produce the same bytes. `tests/test_voice_bank.py` keeps all of that true
against a hand-written `ModelProto` fixture, so the check survives without the
dependency.

The bank layout half is proven the same way: every official and reserved slot is
filled with a marker only that slot should hold and read back at its expected
offset, so "af_heart is still speaker 3" and "the clone in slot 2 is at speaker
55" are numeric claims.

**The runtime half cannot be proven in CI and is therefore enforced at run
time.** Whether the pinned sherpa build loads a derived model and speaks a
custom SID is a question about 350 MB of ONNX and a native library, and no test
fixture can answer it honestly. §55 steps 8–11 are that check, executed on the
user's own machine with their own pinned bundle, every time a bank changes:
install the candidate, start the ordinary worker, synthesize with the proposed
SID, and roll the whole thing back if any of it fails. A clone that cannot be
spoken is never registered, and the previous bank is restored.

That is the honest form of the gate. It is not "we tested it once on a machine
you do not have"; it is "the check runs where the answer is".


## 6. §§58–70 — cloning, and the two places this was narrowed

Both narrowings are the design intent's own rules applied to what actually
exists, and both are visible in the UI rather than papered over.

**One-click installation is not offered, because there is nothing pinned to
offer.** §78 requires a version, a URL, a byte size and a cryptographic checksum
before anything is downloaded. No Model Chain cloning bundle is published, so
`cloning.platforms` in the manifest is deliberately empty and `install()` says
so. The manual path is complete: a validated layout, per-item results, and a
CPU self-check that runs the binary once with the GPU hidden. When a bundle is
published, the manifest gains an entry and nothing else changes.

**Cloning is Linux-only for now** (§59). macOS is excluded because Storytime's
ONNX path uses the CoreML execution provider there, so `--backend onnx` is not a
CPU-only claim anybody should make; Windows waits for a tested bundle. Using a
clone made elsewhere works wherever the ordinary runtime does — making one and
using one are different questions, and the code separates them.

### The process-group lesson

The fake Storytime in the tests forks a child of its own, and it is there
because of a bug the first implementation had. Reading the subprocess's output
until EOF looks correct and is not: a child inherits the write end of that pipe,
so end-of-output does not mean end-of-process. The supervisor sat there after
Storytime had exited, holding a finished clone hostage to a grandchild.

The fix is the shape the rest of the module already wanted. The output is read
on a thread, the *process* is waited for, and when it exits the whole group is
ended — which is both correct containment (§58: nothing survives the job) and
what releases the pipe. `_group` is captured at launch rather than looked up
when needed, because by then the process may be a zombie that has been reaped
and `os.getpgid` raises.

A related detail for anyone writing a containment test: a killed orphan whose
parent has gone is a zombie until something reaps it, and a zombie answers
`kill(pid, 0)` perfectly happily. The tests read `/proc/<pid>/stat` instead.


## 7. §§19–29 — the browser half

`response.body.getReader()` rather than `arrayBuffer()`, an odd trailing byte
carried between reads, `DataView` rather than `Int16Array` (a network chunk does
not promise 2-byte alignment), `AudioBuffer` blocks scheduled against a
`nextStartTime` the page maintains, and a start buffer that rises after an
underrun and relaxes after three clean turns.

The high-water mark is enforced by *not reading*: the socket stops being
drained, the server's send blocks, its queue fills, and the sherpa callback stops
being called. Backpressure to the producer without one dropped sample.

Two things about the composer are worth writing down because they were both
wrong first:

**Hiding Send is only half of it.** When the model finishes, Python's own IDLE
update hides Stop — correctly, as far as Python knows — so a page where Voice is
still speaking needs Gradio's marker taken *off* Stop, not a class of ours added
to Send. `show()` does both directions, and reasserts on every tick and on every
run-state write.

**Busy is a value, not a rendering.** Python publishes `"llm"` or `"idle"` into a
hidden field; the browser holds `voiceBusy`; one function combines them. Reading
visibility off the last CSS Gradio applied is exactly the design that cannot
survive Voice outliving the reply.


## 8. §39 — the microphone policy, stated precisely

The extension's own `isSecureContext` refusal is gone. What replaces it is
capability detection and the browser's own error names mapped to sentences:
`NotAllowedError`, `NotFoundError`, `NotReadableError`/`AbortError`.

What has *not* changed, and is not this extension's to change: mainstream
browsers restrict `getUserMedia` to secure contexts and commonly leave
`navigator.mediaDevices` undefined on an insecure origin. So an Android phone on
`http://192.168.x.x` will usually still not open a microphone. The difference is
what it says — *this browser did not make microphone capture available for this
page* — which is true, rather than a claim about HTTPS that Voice Chat is not
entitled to make. Everything else works over plain HTTP.


## 9. §113 — the migration, and the bug it fixes

The V1 manifest carried `voice: af_heart` beside `speaker_id: 0`. In the
upstream sherpa Kokoro map, speaker 0 is `af_alloy`. Every reply Voice Chat had
ever spoken had been spoken by Alloy.

The manifest now carries the whole 53-name map beside the checksum of the
archive those names came from, `speaker_id` is 3, and the default is stored as
`official:af_heart` — a stable id, never a number, because numbers move when a
bank is rebuilt and a saved default must not. An installation with no stored
voice option gets the corrected default on first read.

The map is validated against the installed model's own `n_speakers` before it is
used. A bundle that disagrees produces no official list and a visible warning
rather than names against numbers that might not be theirs.


## 10. What is deliberately still missing

* Thread counts are unchanged (§35). Streaming was implemented at the
  conservative default first, as the section asks; benchmarking 2/4/6 needs real
  hardware running beside a real image model, and raising a default on anything
  less would be guessing.
* There is no hot bank reload. A bank change stops the worker, installs, and
  starts it again (§57). The optimization that rebuilds only `OfflineTts` while
  Whisper stays resident is explicitly not required for correctness.
* Clones are not resumed across a WebUI restart (§116). An interrupted job says
  *interrupted when the WebUI closed*, which is what happened, rather than
  *failed*, which is not.
