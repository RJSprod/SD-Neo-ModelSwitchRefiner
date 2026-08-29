# Voice Chat responsiveness — implementation notes

Three reports from real use, after the latency work in `15-voice-chat-latency.md`
had landed. Two are bugs; the third is the admission that the second one cannot
be fixed by guessing.

Speech begun immediately after the gesture starts still lost a word. The pause
between the first spoken sentence and the second was still there on some runs.
And Stop, during a Voice-only phase, did nothing at all.

Companion to `13-voice-chat-v1-1.md`, `14-voice-chat-v1-2.md` and
`15-voice-chat-latency.md`. Nothing here changes the architecture those describe:
CPU-only inference, an isolated pinned runtime, no host dependency touched, one
serialized synthesis lane, bounded queues with backpressure, cancellable turns,
no audio persisted, no privilege wanted.


## 1. Stop, and the phase it could not stop

The shortest one, and the most embarrassing. The composer's Stop was built:

```python
gr.Button("Stop", …, visible=False, interactive=False)
```

Hiding a control and disabling it are different statements, and both answers
were "no". `javascript/voice_chat.js` reveals Stop when the language model goes
idle and Voice is still reading the reply aloud — and what it revealed was a
button Gradio had rendered `disabled`. A disabled button dispatches no click
event at all, so not even the capture-phase listener that silences the speaker
locally ever ran. The one moment the user most wants Stop is the one moment it
was inert.

Stop is now built interactive and hidden, and the IDLE update withdraws
visibility without withdrawing enablement. Send keeps its own disable while it
is off screen — a keyboard shortcut aimed at a hidden Send should refuse rather
than quietly send the composer twice — and the browser clears any `disabled` a
re-render puts back, because Gradio may reassert a component's server-side
attributes at any point.

The three cancellation layers are unchanged and still idempotent. The browser
stops scheduled audio and aborts the stream; the Gradio handler closes the
generator through `cancels=` and cancels the Voice turn; the runtime tells the
worker. Duplicate cancellation was always expected here — two paths deliberately
press this button.


## 2. Pre-roll, and what the gesture actually decides

The slide was already being used to prepare the status answer and the
AudioWorklet module. The microphone was not: `getUserMedia` was called when the
handle crossed the engagement threshold, so permission, the device opening, the
stream and the graph all still sat between the end of the gesture and the first
kept sample.

That is a real second on a phone with a Bluetooth headset, and it is exactly the
second in which somebody who has decided to talk starts talking.

So the microphone is asked for in the same task as the press, and the travel is
the window all of that happens in. Samples that arrive before the handle reaches
the end are kept in the page and belong to nobody: the control says "Microphone
open — slide to record", in amber, and the accessible label says the same. The
slide finishing is what makes them an utterance, and the utterance begins at the
first retained sample, because that lead-in is the part the user was worried
about losing.

### This is not a weaker privacy contract

It is worth being precise, because "the microphone opens on touch" sounds like a
retreat and is not one.

The gesture never decided whether the device was touched. It decided whether
anything left the machine, and it still does. An abandoned slide stops every
track, drops every sample, uploads nothing and asks for no transcription. The
browser's own microphone indicator — which is the signal a user actually trusts,
because no web page controls it — comes on at the press and goes off at the
release, and in between nothing has been sent anywhere.

What has changed is that a deliberate slide on a dedicated control is now
treated as consent to open the device, rather than only as consent to send. That
is a smaller step than it reads: the control exists for nothing else, the press
is already a deliberate gesture chosen because no mobile platform wants it, and
the alternative is a feature that loses the first word of every sentence.

Three states, not two:

```
  IDLE          nothing open, nothing buffered
  OPENING       asked for, nothing back yet          amber
  BUFFERING     samples arriving, committed to nobody amber
  RECORDING     the slide finished; those samples are an utterance   red
  WORKING       encoding, sending, transcribing
  ERROR         permission, device, graph or readiness failure
```

### Races, and the token that settles them

Every asynchronous result is checked against the gesture that asked for it. A
`getUserMedia` that resolves after the release is attached to nothing and its
tracks are stopped on arrival; the same rule covers a late readiness answer and
a late module load. One `abandon()` does the whole of section 5.7 — disconnect,
clear ports, stop every track, drop the samples, invalidate the session, reset
the control — because five callers each doing four of those five things is how a
microphone gets left open.

Two checks stay synchronous at the press, because a check that waits for a round
trip has spent the pre-roll it was meant to protect: the model is generating, or
a cached status answer already said Voice is unavailable. Either one refuses
without touching the device. A refusal is *remembered* rather than announced,
and said at the far end of the track — a press that never becomes a slide is not
a request for anything, and a control that complains about being touched is a
control people stop touching.

The assistant's own speaker is silenced at the press rather than at engagement.
A loudspeaker beside an open microphone is the thing most likely to end up in
it.

And a module load that has not finished when the stream arrives is not waited
for. That utterance uses the ScriptProcessor; the worklet finishes for the next
one. Earliest usable PCM beats the preferred graph, and the path is never
swapped mid-utterance.


## 3. Four threads, and the refusal to make that a knob

`TTS_THREADS` moves from 2 to 4. Two was the conservative opening bid, chosen
before there was any evidence about which stage the gap is made of; it is
synthesis, one serialized lane, one sentence at a time, and sherpa's
`num_threads` is what that lane has to work with.

What this deliberately is not is a tuner. Nothing rotates between 2, 4 and 6,
runs an A/B, or selects from real-time factors, underrun counts, CPU load or
prior runs. Two reasons, and the second is the one that matters:

- a production feature that reconfigures itself is a feature whose logs describe
  a different program each time somebody reads them, and the whole of §4 below
  is about making logs comparable;
- the next change to this number should be a person reading evidence and editing
  one line, which is a decision with a name on it.

There is a test whose only job is to make a runtime assignment to that constant
argue with something. Speech-to-text stays at four: a transcription is one burst
after the user has stopped talking, and it was never a stage anybody waited
through.

Process priority is untouched. The worker still asks to lower its own at
start-up and nothing raises it — raising a Linux priority needs `CAP_SYS_NICE`,
and asking a chat feature for a capability is not a trade worth making. What is
new is that the priority is *read* and reported, so a shared log can say whether
the run that produced its numbers was a run at the priority everybody assumes.

That turned out to matter more than expected: the first logs said the read was
failing, which means the write may be failing too and nobody knew. See §6.


## 4. Making one ordinary run answer the question

"There was a four-second pause" is not a report anybody can act on. The previous
release could say how long a turn took and how long the first segment waited;
neither distinguishes "the model was slow to write it" from "Kokoro was slow to
say it" from "the browser's queue emptied", and those want completely different
fixes.

### Per synthesis unit

The worker was already measuring `segment_ms` and the parent was throwing it
away at the reader thread. Each unit now carries four durations, and between
them they name the stage:

```
Voice TTS segment — turn xLYXPq1W, n=2, chars=68, ready_wait_ms=105,
synth_ms=2870, first_block_ms=2812, callback_blocks=1, audio_ms=3120,
streaming=callback
```

Read it as: the text was there in 105 ms, so the segmenter is not the problem;
the synthesis took 2.87 seconds; the first callback came back at 2.81 s of
those, so callback mode delivered nothing early — which for a one-sentence unit
is exactly what §2 of the previous notes predicted; and it produced 3.1 seconds
of speech, so this unit roughly kept up and the one before it did not.

The first two units are always INFO because they are what the reported gap is
made of. Later ones are DEBUG unless they cross a fixed three-second threshold —
fixed, because a threshold that moved with the machine would make two logs
incomparable, which is the one thing this instrumentation exists to avoid.

The turn summary keeps the first two units, the slowest one and its index, the
thread count, the effective priority, the real-time factor and the first-audio
time. A start-up line names the rest:

```
Voice TTS config — threads=4, provider=cpu, streaming=callback,
max_num_sentences=1, priority=nice+5, targets=60/50/100, second_bounds=140/220
```

`max_num_sentences` is on that line because it is the unit `callback_blocks` is
counted in. Four blocks means four sentences only while that is one.

### And the half only the browser knows

Server real-time factors cannot prove the speaker ran dry. A synthesis that kept
up on average still leaves a four-second hole if it fell behind once, and the
only party that knows is the thing scheduling the audio.

The Web Audio scheduler already discovers the moment its next start time is
behind the context clock. How far behind is the length of the silence somebody
heard, and it has to be taken *before* `nextStart` is reset. That, with the
startup buffer and how the turn ended, is posted to a new route:

```
POST /model-chain/voice/telemetry
```

which is deliberately the least powerful route in the module. It reads a fixed
list of numbers and three enumerated words, ignores anything it has never heard
of so a page from a newer build still gets its other fields recorded, clamps
absurd durations, writes one line and returns. Nothing waits for it and nothing
reads its answer: a page that cannot report its own timings has nothing wrong
with its audio. The same route takes the capture summary — how long permission,
the stream, the graph and the first sample each took, how much pre-roll there
was, and whether the recording was sent or discarded.

Every field on it is a duration, a count, or one of a fixed set of words. There
is no field a sentence fits in, and there is a test that checks that against the
declaration rather than against one payload, so a field added later has to argue
with it.


## 5. A size for the second unit, not only a deadline

`SECOND_TARGET` says how soon the second segment may be committed. It says
nothing about how big it may become, and those are not the same question.

A model that writes a hundred and eighty characters before its first full stop
hands the second segment one enormous sentence. It satisfies the target
immediately, arrives as text in a few hundred milliseconds — so the new
`ready_wait_ms` will say the segmenter did its job — and then takes as long to
synthesise as everything it was meant to be covering. Continuity is lost to the
size of the unit rather than to the wait for it.

So the second segment gets the whole envelope early rather than just the target:
a clause or a comma will do at 140 characters, a word boundary has to by 220.
Both are far below the ordinary 320 and 480, and both apply to the final reply
tail as well — a long second unit must not escape the envelope merely because
generation ended before another segment arrived.

Every threshold now comes from one `_limits()` row per unit index rather than
being re-derived at each of the five boundary rules, which is what stops an
envelope being applied on one path and forgotten on another. The limits move;
the guards do not. Decimals, abbreviations, initials, domains, ellipses, fences
and word interiors are not boundaries at 220 characters either.


## 6. What the first logs said, and what was changed because of them

The whole point of §4 was that the next change should be a person reading
evidence. Here is that reading, from one ordinary session: 12 synthesis units
across 6 turns at four threads, plus 15 turns of history at two.

### The four-thread change worked

| | median RTF | sample |
| --- | --- | --- |
| two threads, all history | 0.737 | 13 turns |
| two threads, callback only | 0.877 | 4 turns |
| **four threads** | **0.593** | 4 turns |

Synthesis produces speech about 1.7× faster than it is consumed. Across the six
measured turns there was **one underrun, of 34 ms** — the §11 gate asked for no
gap over 500 ms and zero multi-second gaps, and the worst was thirty-four
milliseconds.

### Time to first audio is one number, and it is the first sentence

Twelve units, least squares:

```
synth_ms = 27.9 × chars + 111
audio_ms = 48.6 × chars + 128
```

`first_block_ms == synth_ms` on ten of the twelve — the one-callback-per-sentence
behaviour, as documented. The two exceptions were multi-sentence units where
callback mode fired early and saved 3.1 s on one of them.

So the budget reads:

```
first audio ≈ 27.9 ms × (characters in the first sentence) + ~250 ms
```

Openings of 28 and 33 characters spoke in 1.07 and 1.17 s. Openings of 84, 85
and ~95 took 2.65, 2.91 and 3.36 s. That is §5's whole justification, and it is
why the opening got an envelope of its own.

### Three things were crossed off, not fixed

**The browser prebuffer is inert.** `first_pcm_to_playback_ms` was 0, 3, 5, 7, 8
and 8 ms on the six turns. One callback block is a whole sentence — 1.6 to 10
seconds of audio — so it satisfies any startup target we would plausibly set.
Lowering it, which §6 of the previous notes deferred, would gain nothing. That
question is closed rather than postponed.

**Capture has nothing left to take.** Engage to first sample was 126–805 ms and
essentially all of it is `getUserMedia`: stream ready at 210–386 ms, graph
assembly 4–45 ms after that. `preroll_ms` was 0 on all six gestures, because the
slide (60–123 ms) finished before the device did. Pre-roll costs nothing and
will help a slower gesture or a slower device; on this machine there was nothing
in front of it to recover. The one thing left was a `sampleRate` constraint that
nothing read, and it is gone.

**Threads stay at four.** Fitting `T(n) = s + p/n` to the 2→4 step gives 6
threads at 1.09–1.19× over 4, with a serial floor implying a ceiling of
1.32–1.92× no matter how many are thrown at it. That is 0.2–0.5 s off a first
utterance for 50% more CPU on every reply — against a first-sentence envelope
worth 0.7–1.1 s for nothing. Continuity does not want it either: RTF 0.59 is
already 70% headroom. It stays a one-line change and `segment_1_ms ÷ chars` is
now the way to settle it, rather than the callback probe, which is confounded by
first-inference warm-up.

### And one thing the logs exposed that nobody was looking for

`priority=class_0x0`. Zero is not a Windows priority class; it is
`GetPriorityClass` reporting failure. `_lower_priority` uses the same undeclared
pointer-sized handle, so the worker's priority may never have been lowered at
all — which means every number above was measured at a priority nobody knows.
The read is corrected and now says "unreadable" instead of inventing a class;
the write is deliberately untouched, because fixing it would make the worker
start yielding CPU for the first time and that is a change to how fast speech is
synthesised, not a diagnostic. The next log will say which case this is.

### Not from the logs, but worth writing down

The 22.7-second wait before the first turn's opening segment was Qwen3 thinking:
llama.cpp shows prompt eval at 1.4 s for 2076 tokens and generation at 90–100
tokens per second. Voice cannot begin until the model emits its first visible
sentence, and with a reasoning model that can be tens of seconds. There is
nothing in this feature to fix there, and the ready-wait field is what says so.


## 7. What is deliberately still missing

* **No lower browser buffer** — and now for a better reason than caution. The
  first measured session says the prebuffer contributes 0–8 ms, because one
  callback block is a whole sentence. There is nothing there to win. See §6.
* **No six threads.** Modelled at 9–19% over four from the 2→4 step, against a
  first-sentence envelope worth several times that for no CPU. See §6.
* **No callback coalescing.** Revision 3 floated combining several Model Chain
  units into one larger `generate()` call to measure per-call overhead.
  Revision 4 removes it from scope: the immediate requirement is diagnosis, it
  would make the block counts above stop mapping to sherpa's batches, and its
  benefit is unproven until the logs show per-call overhead is material. No
  dormant switch is left behind for it, and there is a test that says so.
* **No dynamic priority.** A policy of "normal while producing, lower while
  idle" is plausible and is OS-specific behaviour in a process that currently
  needs no privileges at all. The logs will say whether it is worth the trade.
* **No automatic anything.** Every number collected here is for a person to
  read. The release gate for the next change is a shared log, not a threshold
  crossing.
