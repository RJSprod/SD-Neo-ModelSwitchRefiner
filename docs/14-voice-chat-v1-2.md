# Voice Chat V1.2 — implementation notes

Four changes, and they are not four features. Two of them are the same
observation from opposite ends — that speech quality is something a user has to
be able to *choose*, in and out — and two are the same idea applied to a
character: that a character is not only what it says.

Companion to `13-voice-chat-v1-1.md`, which this does not repeat. Where a
section number appears it is the V1.1 design intent's.


## 1. The Bluetooth microphone, and what is actually wrong

The report: dictation from an Android phone was good on the handset's own
microphone and bad the moment a Bluetooth headset was connected — and "bad" did
not mean misheard words. It meant `(music)`, `(static)`, `[BLANK_AUDIO]`.

That is not a transcription failure. It is Whisper doing exactly what it was
trained to do, and the chain is worth writing down because none of it is a bug
in this extension and all of it is this extension's to survive.

A Bluetooth headset carries audio *out* over A2DP, which is a one-way profile
with no microphone in it at all. Capturing from the headset therefore needs the
hands-free profile, and the phone has to open an SCO link to get it — the same
link a phone call uses. Outside a call, Android decides whether to open that
link at all, and what it opens is narrowband: 8 kHz for plain HFP, 16 kHz where
both ends negotiate mSBC. What reaches the browser is band-limited to about
4 kHz, already through a codec designed for telephony, and usually far quieter
than the handset's own microphone — which the platform gain-stages and
beam-forms itself.

Whisper was trained on 16 kHz speech. Handed a quiet 8 kHz stream upsampled to
look like one, it emits the annotation tokens its training transcripts used for
non-speech passages. `(music)` is the model reporting, accurately, that it could
not hear anybody talking.

So there is nothing to fix and four things to do, and they are in two files.

**In the browser** (`javascript/voice_chat.js`):

* The constraints ask for `sampleRate: 16000` — the rate Whisper wants, rather
  than 48 kHz that has to be thrown away — and the three processors
  (`echoCancellation`, `noiseSuppression`, `autoGainControl`) became `{ideal:}`
  rather than required. They are tuned for a wideband microphone, and a browser
  that cannot apply them to an HFP stream should hand over the stream rather
  than fail the request. A rejected *constraint set* is retried as `{audio:
  true}`, because the devices most likely to reject one are the Android WebViews
  this feature already bends over backwards for.
* The capture path is **read, not guessed**. `track.getSettings()` and the
  track's label say which device this is and at what rate — with
  `getAudioTracks` where the browser has it and `getTracks` where it does not.
  That is the one piece of evidence a user has that their microphone changed
  under them, and it goes to the console on every recording.
* The level is measured, and a quiet recording is **lifted** before it is
  encoded: the gain is chosen so the loudest sample lands at 0.6 of full scale,
  capped at ×12. No compressor and no limiter, which means it cannot clip by
  construction — and a recording that is quiet *because it is mostly silence
  with one loud word in it* is left alone, correctly, because that word was
  already audible.
* A recording still under the floor after the lift is **not sent**. A round trip
  and a large model are a slow way to be told a recording was silent, and the
  answer that comes back is `(music)` rather than nothing.

**On the server** (`mc_voice_hearing.py`, new):

* `measure()` reads peak and RMS out of the WAV before inference. Both floors
  have to be under for a refusal — a recording can be quiet and hold speech, and
  it can hold one pop and no speech.
* `speech()` discards a transcript that is *entirely* one of Whisper's non-speech
  annotations. Anchored at both ends: `(laughs) I said no` keeps its words, and
  so does anybody who dictated the word "music".

Both gates are deliberately conservative and the honest failure mode of both is
to let something through. The one uncomfortable entry is `you` on the filler
list — a real word somebody may say alone, and by a wide margin the single most
common thing Whisper returns for a silent clip. It is discarded, the microphone
is still there to press again, and it is only ever reached by a recording that
already passed the level check.

The messages name the microphone rather than the feature, because that is where
the problem is and the remedy is the user's: speak up, move it, or take the
headset out of the equation.


## 2. Three qualities of speech-to-text

`whisper-base-int8` (Low), `whisper-small-int8` (Medium, the default and what
was already installed), `whisper-medium-int8` (High). All three are the same
publisher's sherpa-onnx exports the existing entry came from.

Two decisions are worth recording.

**Medium stays the default, and this is not a coincidence.** It is what this
installation was already using. An upgrade that silently changed which model
transcribes somebody's dictation would be an upgrade that changed their
transcripts, and "the new version got worse" is not a thing anybody should have
to diagnose.

**`default_id(kind)` was the only function that had to change.** V1's docstring
said so in advance — "asked through here by everything … so that a V2 model
chooser has exactly one function to change" — and it was true. The installer,
the worker's launch paths, `bundle_paths`, the Settings row and the status the
browser reads all follow it. `_status()` had one literal that read the manifest
default directly; that is now the only edit that was not in that one function.

The bundles install into a directory per identifier, which they already did, so
all three can be on disk at once and switching is not a download. Hence two
buttons per card rather than one: **Download** fetches a tier, **Use** points
Voice Chat at it. Choosing the high tier and *then* starting its download is the
order people actually do it in.

Choosing stops the worker. Not because a choice is an uninstall, but because a
loaded worker is holding the *previous* tier's Whisper and the handshake refuses
a worker whose models are not the ones this installation verified (§ the
handshake check). Stopping it here means the next dictation starts a worker on
the new tier instead of failing that check.

### Sizes are approximate and say so

The model bundles are not pinned in this repository — `mc_voice_models` explains
why at length, and it is a maintainer's job on a machine that can reach the
publishers. So the exact download size is not known until `_resolve()` asks the
publisher at install time. The manifest carries `about_bytes` and `ram_bytes`
for display only; they participate in no verification, and the row says
"approximate" rather than presenting a figure that could be wrong by megabytes
as though it were exact.

The RAM figures are the other half of the choice and are the reason the cards
exist at all. The high tier holds roughly three gigabytes while it is loaded, on
top of whatever the language and image models are already using, on a machine
whose whole point is running those at the same time.


## 3. Delivery, and what Kokoro actually offers

This is the section that would otherwise be discovered by somebody adding an
"emotion" slider.

`sherpa_onnx.OfflineTts.generate` takes exactly two things that change the
sound: `sid` and `speed`. There is no emotion input, no style vector beyond the
speaker's own, no per-utterance prosody conditioning, and no SSML. The 1×256
style vector the graph is handed is selected by speaker id and token length and
is otherwise fixed — `mc_voice_bank` documents the same arithmetic from the
other side.

So one of the four controls is the model's and three are ours, and
`mc_voice_profile.py` says which is which in its docstring rather than leaving
it to be inferred:

| control | whose | how |
| --- | --- | --- |
| Speed | Kokoro's | straight into `generate` |
| Pitch | ours | synthesize at `speed × ratio`, resample the result by `ratio` |
| Volume | ours | a scalar folded into the PCM16 conversion |
| Pacing | ours | silence between segments |

**Pitch is the interesting one and it is honest about how it works.** Resampling
by a ratio shifts every frequency by that ratio *and* divides the duration by
it, so the synthesis speed is multiplied by the same ratio to put the length
back where the speed setting asked for it. Formants move with the pitch, so this
reads as a different-sized speaker rather than as a transposed one — which is
exactly what makes it useful for giving two characters the same voice and
different bodies, and exactly why an octave sounds like a cartoon.

Two implementation notes on the resampler:

* **One instance per segment, and the read position is carried between blocks.**
  sherpa's callback delivers batches at whatever granularity it chooses, and an
  output sample routinely needs two input samples that arrived in different
  batches. Restarting at each block is a click at every boundary.
* **The tail is dropped as it is consumed.** Everything before the next read
  position can never be read again, and a reply that kept its first sentence
  would be a memory leak with a sample rate.

The pure-Python cost is real and bounded: linear interpolation over 24 kHz
audio, in a runtime with no NumPy, is a percent or so against an RTF already
around 0.7 — and it is skipped entirely when the numbers are neutral.
`Delivery.shapes` is false for the default profile, so an installation that
never opens these controls runs the path it ran before they existed.

**Pacing does nothing on the `tts` route and this is not quietly approximated.**
That route is one `generate` call with no segment boundaries in it. The only
text that goes down it is an audition or a single short fallback reply, so the
control that inserts silence *between* sentences has nowhere to put any.


## 4. A character's own voice

The voice is stored in the character file, beside its sampling, as the stable id
`mc_voice_registry` mints — `official:af_nicole`, never a speaker number, for
the reason that module gives at length: a number is an address in a block of
floats and moves when a bank is rebuilt.

### Inheritance is by absence

A delivery field that is `None` means "follow the default voice", and that is
the value rather than a missing one. The alternative — writing today's defaults
into every character — is the one where somebody slows the default voice down
and the characters they never configured do not follow. So `to_mapping` writes
the voice keys only when they hold something, a character with four `null`s in
it would read as configured, and `mc_voice_profile.resolve` is the one place the
merge happens.

In the editor that is one checkbox over four sliders. The sliders open showing
the *effective* values — the default's, for a character that has none — so they
start where the sound the user is listening to actually is.

### Why the list is painted rather than built

The compact picker in the character screen is fetched from `/voice/voices`, the
same route the Settings list uses, for the same reason: the list changes when a
clone finishes and when a voice is renamed, and a Gradio dropdown rebuilt from
Python is a list that goes stale the moment either happens.

What Gradio owns is the *value*. The browser writes the chosen id into a hidden
textbox and the ordinary **Save character** reads it with everything else — no
second save, no second store, and nothing to get out of step with the character
file. The highlight follows that field on a one-second DOM read rather than on
an event, because Gradio does not tell this file when Python writes into a
component.


## 5. The bug this found on the way past

`speak()` — the completed-reply route, the non-streaming fallback — called
`runtime.synthesize(text)` with no speaker at all. sherpa answers an absent
speaker with speaker 0, which in the upstream Kokoro map is `af_alloy`.

That is section 113's bug, still live in the one path nobody had looked at when
the rest of it was fixed. It mattered less then, because it is the fallback. It
matters now: a character with a voice of its own would have kept it on every
streamed reply and lost it on exactly the replies that could not stream.

So a remembered reply carries *how* it is to be spoken alongside *what* — which
is the same immutable-snapshot rule R2-5 already applies to the words, for the
same reason. Which character was talking is a question only the moment the reply
completed is certain of the answer to, and `speech_marker` now takes the panel's
own character reader as a closure so that the dependency still points one way.


## 6. What is deliberately still missing

* **No emotion control.** Kokoro-82M has no emotion input. A slider for one
  would do nothing, and `tests/test_voice_profile.py` has a test whose only job
  is to make adding one argue with something.
* **No per-character speech-to-text.** Which transcriber is listening is a
  property of the machine, not of who is being talked to.
* **Pitch is resampling, not a vocoder.** A phase vocoder would preserve
  formants and would be a signal-processing dependency inside a two-wheel
  runtime. The formant shift is a feature here rather than an artefact — see
  §3 — and at the few semitones anybody actually uses it is what makes the
  control worth having.
* **The tiers are not benchmarked here.** The notes on each card describe
  accuracy and speed in relative terms, because measuring them honestly needs
  the user's own CPU with their own image model running beside it, which is the
  same reason §35's thread counts are still at their conservative default.
