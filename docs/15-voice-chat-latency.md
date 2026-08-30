# Voice Chat latency — implementation notes

Two complaints, and they turn out to be the same complaint told from both ends
of the pipeline.

Automatic speech began later than it needed to, and there was a conspicuous
silence between the first spoken sentence and the second, after which playback
was generally continuous. And the microphone slider reached its far-right red
state before any audio was being captured — by about a second on a phone with a
Bluetooth headset.

Neither is a bug in one function. Both are a chain of stages that were serial
when they could have overlapped, plus one piece of interface that was reporting
an intention as though it were a measurement.

Companion to `13-voice-chat-v1-1.md` and `14-voice-chat-v1-2.md`, which this
does not repeat. Nothing here changes the architecture those describe: the
speech runtime is still isolated, still CPU-only, still provisioned from a
checked-in pinned closure; the worker is still lazy and still contained; queues
are still bounded and still apply backpressure rather than dropping; the
microphone is still opened only by a deliberate gesture and never held between
utterances.


## 1. What the first-to-second gap actually was

The shape of the report is the diagnosis. *First sentence, gap, then fine.*

Three things produced exactly that shape, and only the third is the one everyone
guesses.

**The opening was held for the wrong reason.** The segmenter is allowed to
commit a short complete first sentence early — "Yes, that's possible." is the
whole answer, and waiting for the paragraph after it would be absurd. But the
condition was that the sentence also had to be the end of everything buffered.
A model that emitted `Yes, that's possible. Here is the` in one delta had
finished the first sentence at exactly the moment one that emitted the sentence
alone did, and the first was held while the second was spoken. The first
segment's latency was a property of chunk boundaries.

**The second segment had no lead and the ordinary target.** Segment one was
already playing. Segment two had to reach a hundred characters of a reply that
was still being written, and a hundred characters can be several seconds away.
Playback of sentence one ran out before sentence two existed. By segment three
there was audio queued ahead of it, which is why the gap did not repeat — the
reported shape, exactly.

**And on a cold run, the worker started too late.** The turn pump waited for the
first committed segment and only then opened the turn, which is what ultimately
starts the worker. So four hundred megabytes of ONNX were read *after* the model
had finished its first sentence rather than while it was writing it. Both are
the machine waiting; only one of them is the user waiting.

To which the browser added its own two: a 400 ms poll to notice the reply
existed, and a 1.6-second prebuffer before playing any of it.


## 2. NumPy, and what callback streaming is really worth

The speech runtime is unpacked wheels in an interpreter of its own, and it did
not have NumPy. sherpa hands a batch of samples back mid-synthesis through a
pybind11 `py::array_t<float>`, which pybind11 cannot build without NumPy in the
interpreter running the callback — so the callback raised, the worker fell back
to segment streaming, and every spoken reply waited for a whole `generate()`
call to return. (The completed-audio path was unaffected, because `samples`
comes back as a plain list. That is why auditions worked while every streamed
reply took the slow path.)

So NumPy joins the closure. Not the host's NumPy: Forge has its own, with its
own constraints and its own dependents, and Voice Chat has no business having an
opinion about it. NumPy 2.2.6 is pinned for all eight supported platform/Python
combinations in `voice/managed-voice-models.json`, fetched and hashed by the same
code path as the engine, and unpacked into the same isolated interpreter.

The version was chosen against the runtime rather than for being recent.
sherpa-onnx 1.13.6's compiled module contains both the `numpy.core` and
`numpy._core` lookup strings and a pybind11 recent enough to carry
`pybind11_conduit_v1_`, so it is a build that expects to meet NumPy 2; 2.2.x is
the last line that still publishes cp310 wheels, which the closure needs. Built
from the pinned wheels and run, the isolated interpreter imports both and the
worker self-test reports sherpa-onnx 1.13.6 with NumPy 2.2.6.

### What it buys, measured

This is the part worth being careful about, because "callback streaming" sounds
like sub-sentence audio and is not.

sherpa calls the generation callback once every `max_num_sentences` sentences,
and this worker asks for one. On the pinned runtime, with a small VITS model
standing in for Kokoro:

| request | blocks | first block | whole synthesis |
| --- | --- | --- | --- |
| one sentence | 1 | 122 ms | 122 ms |
| four sentences | 4 | 64 ms | 286 ms |

So callback mode overlaps production *across* sentences — a four-sentence
request hands its first audio over 222 ms before `generate()` returns — and does
nothing whatever inside one. A segment containing a single sentence gets its one
block at the moment the synthesis is already finished.

With NumPy removed from that same runtime, the callback raises
`ModuleNotFoundError` before the first batch and the completed path still
returns a list of samples: the failure the worker's probe was written for,
reproduced.

That is the whole reason §3 and §4 below exist as separate work. A capability
probe that says "callback" proves callbacks work; it does not prove any audio
arrived early. The worker therefore reports how many batches each segment took
and how soon the first arrived, and the turn carries both — so the claim can be
checked in a log rather than argued about.

### Making an existing installation notice

Adding a wheel to the manifest changes nothing on a machine that already has the
runtime. Freshness was decided from sherpa's version and the platform id, and
neither of those moved: an installation from the previous release would have
reported itself current while missing the one wheel this release adds.

There is now a build number for humans and, beside it, a fingerprint derived
from the platform id and the ordered `name:sha256` of every artifact. The
fingerprint is what freshness is actually decided on, because it cannot be
forgotten — add a wheel, remove one, reorder the closure or re-pin any single
hash, and it changes without anybody having to remember that it should.


## 3. Producer continuity

Two changes to `mc_voice_segment`, both about *when* a boundary is taken and
neither about *where* one may be. Every guard is untouched: decimals,
abbreviations, initials, domains, ellipses, markdown fences and the label
`clean_reply` strips are all still boundaries this module refuses.

**The opening takes the earliest validated sentence end** that is long enough to
be a sentence, whether or not more text has arrived behind it. `SHORT_SENTENCE`
is what stops that from turning `1.` into a segment, and it is why "Sure." is
still not spoken on its own.

**`SECOND_TARGET`, at 50 characters**, sits between the first segment's 60 and
the ordinary 100. Only segment two is hurried, and the reason is stated in the
code: it is the one with nothing queued in front of it. Applying a target that
low to every segment would turn one synthesis call into four and put a seam
between each of them.

**The worker warms up on the turn thread**, before the first segment is waited
for. `mc_voice_runtime.prepare()` is `ensure_started()` with a question in front
of it — it registers no turn, claims no inference lane, sends no `tts_begin` and
synthesizes nothing, so a warmed worker that is then cancelled leaves behind
exactly what the first real use would have left. `add_text` is unaffected: it
runs on the generator's thread and the warmup runs on the turn's.

A cold-start failure is deliberately left for `begin_turn` to report through the
path that already exists for it. Cancelling the turn in `prepare` would describe
a worker problem as though the reply itself had gone wrong.


## 4. The browser

**Noticing the reply.** Gradio writes the hidden turn field and dispatches
`input` on it, so there is a signal to listen to and the poll was 0–400 ms of
latency paid for a change the page could have been told about. Gradio also
*replaces* those controls when it re-renders, and a listener on a node that has
left the document will never fire again — so an observer on the holder re-binds
when that happens, idempotently, because a second listener on a surviving
control would speak one reply twice.

The poll stays exactly where it was, and this is the part worth writing down:
`value` is an IDL property. Assigning it fires no event and changes no attribute
a `MutationObserver` is watching, so a theme or a future component that writes
it directly would be invisible to both of the mechanisms above. The poll is the
recovery path, not the mechanism.

**The startup buffer** comes down from 1.6 s to 0.7, floor 1.2 → 0.4, ceiling
3.2 → 2.0. After the producer-side work, not before it: a smaller buffer in
front of a producer that runs dry is a feature that starts sooner and stutters,
and the goal is earlier *continuous* audio. It is still adaptive in both
directions — an underrun raises it, three clean turns relax it a step — and it
is nowhere near zero.

One subtlety fixed on the way past: a turn now keeps the target it started with.
Reading the shared value while blocks are arriving meant an underrun could
deepen the queue of the very sentence that was already playing.

### What a head start cannot buy

Added after Sopro V2 shipped, because Sopro on CPU is the first engine here that
can run *slower than real time* and a startup buffer turns out to be the wrong
instrument for that entirely.

A fixed head start hides a bounded shortfall. When the producer is slower than
real time the shortfall is not bounded — it grows for every second the reply
lasts — so no prebuffer fixes it, and sizing one for the longest possible reply
would tax every short one. A reported case: RTF 1.16, a 38-second reply, six
seconds of silence owed. The ceiling of 2.0 s covers about fourteen seconds of
that reply and nothing after it.

What *was* fixable is the shape of the failure. The old code resumed on the first
block to arrive after the queue emptied, with 20 ms of lead — which guarantees
the block after it is late too. One shortfall became a rattle for the rest of the
turn: **56 separate silences** in that reply, averaging a twentieth of a second
each, which is heard as a broken speaker rather than a slow one.

So a dry queue is now treated as a rebuffer. Blocks are held until there is a
second of them, the hold doubles each time it proves too short, and it stops at
six. The listener gets a handful of pauses between sentences instead of a
continuous stutter, and on a machine only slightly behind, the first pause is the
only one. `rebuffer_count` and `rebuffer_target_ms` go into the playback report
beside `underrun_count`, because "the speaker ran dry" and "this page chose to
stay quiet and refill" are different events and a turn where they track each
other is a producer that cannot keep up at all.

This is the one adjustment deliberately *not* deferred to the next turn. The
startup buffer is, for the reason two paragraphs up. A rebuffer is the opposite
case: the speaker has already fallen silent, and the only question left is
whether it resumes into another gap.

### The envelope learns from every ending

The adaptation above was reached only from `finishSpeech`, so it ran only when a
stream reached its end. A turn the listener stopped taught it nothing — and the
turn a listener stops is the turn that stuttered. A log from the machine in the
case above showed the same 700 ms head start on three consecutive turns with 10,
56 and 11 underruns between them, because every one of those turns was cancelled
before it could report.

Raising and relaxing are not symmetrical, though, and the fix has to keep that
straight. Underruns are evidence however the turn ended: the speaker did fall
silent and the listener did hear it. A clean run is evidence only if it was
allowed to finish — a reply stopped two seconds in has not shown that anything
works, and letting it relax the target would let a user who interrupts a lot talk
the page into a head start too short for the first reply they actually listen to.
So: raise on any ending, relax only on a clean one.


## 5. The microphone, and what "recording" is a claim about

The old sequence was: the slide reaches the end, the control turns red, and then
`getUserMedia`, the audio graph and a status round trip to the WebUI all happen.
On a phone opening a Bluetooth headset that is about a second of red before any
audio exists.

It was not only cosmetic. The capture callbacks began with

```js
if (!capture || capture !== state) return;
```

and `capture` was assigned only after the status request came back — so every
frame that arrived first was dropped. Audio the microphone really had captured,
thrown away because an HTTP request was in flight.

**There are two states now.** Engagement enters OPENING: the handle pins at the
right, the track goes solid amber, the label says "Opening microphone".
RECORDING — the red — begins at the first PCM frame the page accepts, from
either capture path, through one `acceptChunk` helper so the two cannot drift
apart. The text is authoritative for assistive technology, as it already was;
the colour follows it.

**The clocks moved with the state.** The minimum hold and the sixty-second cap
are both counted from the first sample, so a device that was slow to wake spends
its own time and not the user's — and 800 ms of opening plus 80 ms of speech,
which used to read as a comfortable 880 ms recording, is now correctly a tap. A
separate opening timeout gives the control back if the microphone never opens at
all, which is a different failure bounding a different thing.

**Releasing during OPENING** cancels the session, stops whatever tracks arrive
afterwards, never turns red, never transcribes, and says nothing about the
recording being too short — because there was no recording, and the wait was the
device's.

**Sample acceptance no longer waits for the status route.** What the answer still
decides is whether the recording is *kept*: a late "not ready" stops the tracks,
discards the samples, resets the control and shows the route's own sentence. A
readiness that is already known to be false still refuses before any microphone
is opened, because a permission prompt raised for a feature that cannot run is a
prompt asked for nothing.

**The capture processor is registered once per AudioContext.**
`registerProcessor` puts a name in the context's `AudioWorkletGlobalScope`, not
in a recording, so calling `addModule` again for each utterance was a duplicate
registration — which a browser may refuse outright, and which is in any case a
Blob, an object URL and a module load per press. One preparation per context,
remembered; a fresh `AudioWorkletNode` per utterance, which is what a node is
for. The preparation starts during the slide, which touches no microphone.

And it is never awaited before `getUserMedia`. A browser raises a permission
prompt only while it still considers a user gesture in progress, and a promise
chain that has awaited a module load is past that on a phone. Both start in the
same task; the graph picks the worklet up if it is ready and falls back to the
ScriptProcessor if the load genuinely failed.


## 6. Clocks

Every duration the browser measures now comes from `performance.now()`. The wall
clock moves when a machine synchronises time or a laptop wakes, and a recording
held for 300 ms across such a jump could be measured as a negative length — which
the minimum-hold guard would have read as a tap.

Server durations stay on `time.monotonic()`. The two are never subtracted from
one another: two monotonic clocks in two processes share no origin, and the
difference between them is a number that looks like a latency and is not one.
Cross-layer analysis is done with local durations and ordered events.


## 7. What is written down, and what is not

Content-free, both sides, and named rather than filtered — a field is absent
from a log until somebody puts it there deliberately.

Server, per turn: whether the worker was warm at the start, how long warming
took, whether speech was produced slower than it plays and by how many seconds
of owed silence, the first two segments' character counts, how long the synthesis lane
stood idle between segment one and segment two, which way the worker delivered,
the block counts for the first two segments and how soon the first block
arrived. The block count doubles as the sentence count, because
`max_num_sentences` is one.

Browser, per turn: how long after the turn was seen the stream opened, the first
sample arrived and playback began, the startup buffer used, the underrun count
and the total and longest gaps, and how many times the page rebuffered and what
hold it had reached. Per recording: milliseconds from the engagement to the worklet, the
microphone, the graph, the readiness answer, the first sample and the release —
which is what separates a slow permission prompt from a slow module load from a
device that simply takes a second to wake.

No text, no transcript, no samples, no turn id, no device name in the timing
line. A configured `pause_ms` is excluded from a segment's first-block time,
because prosody somebody asked for is not slow synthesis.


## 8. What is deliberately still missing

* **No thread-count or process-priority change.** `TTS_THREADS` is 2 and the
  worker still lowers its own priority, both so that speech can coexist with an
  image being rendered. They are worth benchmarking now that the pipeline is
  not the bottleneck, and benchmarking them honestly needs a real desktop CPU
  with Forge generating beside it — the same reason §35's defaults have not
  moved since V1.1.
* **No adaptive segmentation.** Passing the worker's streaming capability into
  the segmenter — bigger segments in callback mode, more eager commits in
  segment mode — is technically sound and is cross-layer policy coupling. The
  fixed second target is the cheaper thing to try first, and it should be shown
  to be inadequate before that coupling is introduced.
* **No claim that callback mode improved one-sentence first-audio time.** It
  did not, it cannot, and §2 has the measurement. What it improves is producer
  continuity across a multi-sentence segment.
* **The buffer constants are a starting envelope, not a verdict.** 0.7 / 0.4 /
  2.0 is well inside the range the design intent asks for and is the kind of
  number that should be re-chosen from measured p50 and p95 startup and underrun
  behaviour on real machines. The page already reports both.
