# Voice Box engines — Breeze TTS 2 beside VibeVoice 7B

The Voice Box (docs/23) renders a script with one engine at a time: **VibeVoice
7B**, as it always has, or **Breeze TTS 2**. This file is the design: how the
Voice Box reaches an engine, what Breeze adds, what was settled with the owner,
and what has not run yet. Behaviour people see is in `README.md` ("Voice Box");
the reasoning behind a line of code is in that module's docstring.

The brief was the owner's *Breeze TTS 2 in the Voice Box* (2026-10-14), asking
for the whole scope in one pull request rather than four, and for the community
Flash attention 2 wheel to be installed in Breeze's own runtime.

---

## 1. Decisions recorded with the owner

| Question | Decision |
|---|---|
| CFG | **4 by default, on the lines it applies to**: a line with a direction (its own or its slot's) and a designed voice. A line spoken plainly in a cloned voice is always at 1, which is all upstream's cloning template accepts (it has no negative prompt) |
| The gap between two speakers' lines | **A configuration field**, *Line gap (ms)*, 0–2000, 150 by default. None within one speaker's run |
| Flash attention 2 | **Installed with Breeze's runtime** from the community wheel for CPython 3.13, torch 2.9.1 and CUDA 12.8, and **the default attention** (section 5 says what it covers) |
| Reuse settings across engines | **Restore the shared fields, and name the engine** that made the output; the engine is never switched for you |
| Pull requests | One, with the whole scope (the brief planned four) |

---

## 2. The seam

### 2.1 The registry: `mc_voice_box_engines.py`

One frozen row per engine, as Voice Chat's `mc_voice_engines` has (docs/18):

| id | label | guest (`mc_turns`) | adapter | runtime |
|---|---|---|---|---|
| `vibevoice` | VibeVoice | VibeVoice | `mc_voice_vibevoice` | `mc_voice_vibevoice_runtime` |
| `breeze` | Breeze TTS 2 | Breeze | `mc_voice_breeze` | `mc_voice_breeze_runtime` |

`active_id()` reads the Voice Box's `engine` setting (VibeVoice when none is
recorded). `select(engine, client)` is the only way it changes: refused with 409
while any render or design is queued or running; otherwise the outgoing engine's
workers are ended on every card — through the turn client, so a warm stay ends
the way it always ends (the parked image model comes back, WanGP's lease is
released) — then by the runtime's own `shutdown`, and only then is the choice
written. Nothing of the incoming engine is started: its worker starts on the next
render. So at most one engine's worker is ever resident, and the turn system sees
exactly what it saw with one engine. `shutdowns()` hands the WebUI's shutdown door
every engine's `shutdown`.

The Voice Box service names no engine: `mc_voice_box.py` holds no `vibevoice`
and imports no engine's module (`tests/test_voice_box_engines.py`). The one place
the first engine is special is backward compatibility — VibeVoice's options as
configurations and outputs held them flat before there were engines — and that
is the registry's `DEFAULT_ENGINE` and the adapter's own `FLAT`.

### 2.2 The adapter contract

What the service, its routes and the page read off an adapter: `ID`, `LABEL`,
`GUEST`, `MAX_TAKES`, `SETTINGS_DEFAULTS`/`SETTINGS_KEYS`, status, settings,
install, `need_vram_bytes`/`need_ram_bytes`/`note_peak`, `refusal`, and the
engine's **vocabulary as data**:

* `FIELDS` — the configuration fields, in drawing order: `key`, `label`,
  `kind` (`int`, `float`, `bool`, `choice`), bounds and step, `blank` (a
  nullable int's placeholder), `default`, `options` for a choice, `needs` (a
  switch it means nothing without), `place` (`head`: after Card, before Seed and
  Batch; `tail`: after them), `wide`, `hint` (a tooltip only) and `infotext`.
  `mc_voice_box_fields` validates against them generically; `fields()` is the
  list as the page draws it, with an attention the runtime cannot run marked
  unavailable and why.
* `clean(values, previous)` — one configuration's options, checked, in the
  Voice Box's sentences.
* `refuse(options)` — why these options cannot render on this machine (Flash
  attention 2 without the package), said at the press.
* `infotext(render, shared)` — the parameters of an output's line, in order.
* `capabilities()` — `speakers` (`native` or `per_line`), `takes` (`batch` or
  `sequential`), `design`, `directions`, `voice` (`sample`, `transcript`,
  `description`, `instruction`), `events` (the vocal events the engine performs,
  or none) and `languages`. The page draws Breeze's extras only while these say
  so, so VibeVoice's page is what it was.

### 2.3 The runtime contract and the neutral job

`mc_voice_box_render.py` holds what the service hands a runtime and gets back,
engine-neutral: a `RenderJob` per section — `lines` (`Line(speaker, text,
instruction)`), `voices` (`Voice(pcm, rate, sample_id, title, transcript,
description, instruction)`), `seed`, `takes`, `options` (the engine's own,
cleaned) — and a `RenderResult` of `Take`s. A runtime offers `load(card,
options)`, `render(card, job, on_progress)`, `cancel`, `unload`, `evict`,
`status`, `rendering`, `resident_bytes`, `shutdown`, `stop_on_exit`. VibeVoice's
runtime translates the neutral job into the header its worker always read (its
worker is unchanged); a line's direction is dropped there, so a script moves
between engines without edits.

### 2.4 Guests and turns

`mc_turns_guests` registers one guest per row (`EngineGuest`, which looks the
runtime up by name on every call). The turn client's `request` and `unload` take
the guest's name; the active engine's when none is given. `mc_turns`' sentences
name the guest by its label — *Waiting for Breeze TTS 2 on the RTX 3090 — this
generation starts when it is done* — and its settings are worded for the Voice
Box rather than for VibeVoice. The broker's voice family is called *Voice Box*.

---

## 3. Settings, configurations and outputs

* **Settings** gain `engine` (VibeVoice by default), changed only by `/engine`;
  `/settings` never switches it. The card, the model and the warm stay are
  shared and mirrored into the active engine's own settings file; a model id of
  the other engine's is not told to the active one.
* **A configuration** keeps every engine's options under `options` by engine id
  (defaults where an engine was never saved), `engine` (the engine it was last
  saved for), `engines_saved`, and `slots` — per speaker, a standing
  `instruction` and a `description` to design from. VibeVoice's options are
  still written flat beside them, and a configuration saved before engines is
  VibeVoice's. A body with flat keys and no options map is an older page's
  VibeVoice options; a flat value that differs from the body's own options map
  is such a page's edit and wins (the page never sends flat keys).
* **An output's record** gains `engine` and `options`; a VibeVoice render keeps
  writing its flat keys. `render.configuration` (Reuse settings) gains
  `engine`, `options` (that engine's) and `slots`. An origin from another tab
  gains `engine`, the one that answered.
* **The infotext** of a VibeVoice output is byte for byte what it was (held by a
  test against records written before). Any other engine's line starts
  `Engine: Breeze TTS 2, ` and lists its own fields.

---

## 4. How Breeze renders a section

Breeze is a single-speaker model, so a section is rendered **line by line**, each
line in its speaker's voice, and joined: the configuration's *Line gap* between
two lines of different speakers, nothing between one speaker's lines. Takes are
rendered **one after another**, the whole section per take, take *k* at seed +
*k* (every call reseeds with `set_all_seeds` and the streaming runtime's own
seed). A voice is:

* **a sample with its transcript** — cloned with `ref_clone_tata` at CFG 1; or,
  with a direction (the line's own, else the slot's standing one), with
  `ref_edit_tata` at the configuration's CFG;
* **a description** (a voice being designed) — `tts_instruction`, the
  description as the instruction and a line's direction after it, at the
  configuration's CFG. Every call samples a new voice from a description, which
  is why a described voice is designed once and kept (section 6).

Each speaker's sample is encoded to codec frames **once per render**, however
many lines and takes use it, and no file is written: while a render is open the
worker answers upstream's `templates._encode_prompt_audio` for its own keys from
those frames. A line longer than one call allows (1,500 codec frames, two minutes
at 12.5 a second, and the 2,048-token sequence) is split at its sentence ends,
then clauses, then words, every piece cloned from the same reference; nothing is
dropped or reordered. A call that fills its room is reported (`capped`). The
cancel flag is read at every chunk the codec hands back (160 ms of audio on the
eager path). Progress frames carry the seconds made in the take, the line and
the take, and the status line shows *line n of m, take k of t*.

**Sampling** is on by default because upstream's generation configuration
samples (temperature 0.9, top-p 1.0, top-k 50, for both the backbone and the
depth decoder); the configuration's temperature and top-p override both, and
off is greedy, which upstream never uses. The repetition penalty stays at
upstream's 1.1. Upstream's separate reference and instruction guidance scales
are left at none, as its own command line leaves them.

---

## 5. Attention, Flash attention 2 and the fast path

transformers 4.57's Flash attention path reads any attention mask as a 2-D
padding mask, and Breeze's two decoders hand it a 4-D additive mask; so Flash
attention 2 can only ever run Breeze's **text encoder** (T5Gemma 2), which reads
`_attn_implementation` from its configuration at every forward pass. The three
choices are therefore:

| Choice | Text encoder | Backbone and depth decoder |
|---|---|---|
| **Flash attention 2** (default) | Flash attention 2 | Eager |
| **SDPA** | SDPA | SDPA |
| **Eager** (upstream's) | Eager | Eager |

Flash attention 2 and Eager build the same model, so switching between them is
one attribute; SDPA builds the decoders differently and is a reload. The worker
refuses Flash attention 2 when the package is not importable; the installer's
self-test *imports* it (a wheel built for another torch fails exactly there) and
records whether it loads, and until it does the choice is listed greyed — *not
installed*, or *installed, but it would not load* — and refused at Render.

**The fast path** is an engine setting (`fast_path`, off), not a configuration
field. Upstream's fast path is CUDA graphs for the text encoder and the backbone's
decode, and `torch.compile` for the depth decoder and the codec. Windows has no
Triton, so `torch.compile` has no backend there: the worker turns the compiled
stages on only when Triton is importable, the prefill graphs off (they are
frozen to a 256-token prefill by upstream's warm-up profile, and a cloned line's
prompt is longer), and uses no frozen profile. On the owner's machine the fast
path is therefore the two CUDA-graph stages, and the log says *fast path (CUDA
graphs only)*. A warm worker keeps the path it was loaded with until it leaves
the card.

**Memory.** Upstream states about 7.7 GiB for eager inference and 14.4 GiB with
every fast stage. A turn asks for the weights (the installed files' sizes once
there are some) and a 1 GiB working set, with 7.2 GB more while the fast path is
on; takes do not add, since they run one after another. Every render's peak is
kept per path (`<model>` and `<model>+fast` in Breeze's `calibration.json`), and
the larger of the estimate and the measurement is what the turn asks for.

---

## 6. Breeze's own

* **Transcripts on samples.** A sample keeps its exact words (`transcript`, up
  to 2,000 characters, and `transcript_source`: typed, whisper or design) in its
  own record beside its sound, and nowhere else. With Breeze active a speaker
  whose sample has none refuses Render at the press, naming the sample.
  **Transcribe** fills the box with Voice Chat's Whisper — the same recognizer,
  in its CPU worker, whatever Voice Chat engine is selected, once Voice Chat's
  models are installed — and saves nothing: the words are checked and saved by
  the person.
* **Whisper hears sixty seconds.** sherpa-onnx's Whisper decodes the first
  thirty seconds of a stream and discards the rest (its own log line), so a
  sixty-second sample — or a sixty-second dictation in Voice Chat, which had the
  same fault — came back as its first half. Voice Chat's worker now decodes a
  longer recording in pieces of at most 28 seconds, each cut at the quietest
  20 ms of its last four seconds, and joins the readings (no space between CJK
  characters); a piece that heard nothing — an annotation, a line of marks or
  one of Whisper's sign-offs, by the parent's own rules copied into the worker
  and held equal by a test — is left out.
* **Voice design.** A slot can hold a description. **Design** renders one take of
  a fixed sentence (English, or Chinese for a mostly Chinese description, or the
  person's own) in a voice made from it, with the configuration's card,
  attention, sampling and CFG, at the configuration's seed or the next one for
  **Redraw**; nothing is made in any pipeline. **Keep** makes the take a sample
  (`source: design`) whose transcript is the sentence it spoke and whose
  `design` record keeps the description, seed, CFG and sentence; the slot then
  points at it, and every line after is a clone of it.
* **Directions.** A slot's standing `instruction`, and a line's own:
  `Speaker 2 (whispering, afraid): I heard something.` The parser reads the
  optional group for both engines; VibeVoice drops it.
* **Vocal events.** `(laugh)`, `(sigh)`, `(cough)`, `(clears throat)` and
  `[笑]`, `[叹气]`, `[咳嗽]`, `[清嗓子]` pass through as words; Breeze performs
  them, VibeVoice reads them.

---

## 7. Breeze's runtime and its trust root

`voice/managed-breeze-models.json`, written by `tools/pin_breeze_models.py`
(whose solver is VibeVoice's pin tool's), for `windows-x86_64-cp313-cu128`:

* **The PyPI closure**, fifty wheels named, sized and hashed from pypi.org,
  solved from `qwen-tts` 0.1.1 and the `transformers` 4.57.3 and `accelerate`
  1.12.0 it pins, with Breeze's floors for `huggingface-hub`, `numpy` and
  `soundfile`. `qwen_tts` imports `librosa`, `sox`, `onnxruntime` and `einops`
  when Breeze's loader imports its codec, so they and theirs ship; its demo's
  `gradio` does not, nor `hf-xet` (behind a marker Windows does not match), nor
  Breeze's HTTP server and tests.
* **torch and torchaudio 2.9.1** from the CUDA 12.8 index, versioned resolves
  answered on the owner's machine against the digests the index states (the
  machine that wrote the manifest cannot reach download.pytorch.org).
* **Flash attention 2**: `flash_attn-2.8.3+cu128torch2.9.1cxx11abiTRUE-cp313-
  cp313-win_amd64.whl` from Wildminder's AI-windows-whl, a declared download
  checked against the digest its host states and recorded in the untracked
  overlay.
* **Breeze's own code**: not a wheel, and a GitHub archive's bytes are made on
  request, so the repository's files at commit `6eba901d1845…` are pinned one by
  one from raw.githubusercontent.com with their SHA-256s — the four
  `breeze_infer` modules the worker imports, the whole `models` package,
  `configs/fast.json` and the licence — and installed under
  `runtime/breeze-tts/` with a `COMMIT` file, which the worker puts on its path
  through `MODEL_CHAIN_BREEZE_SOURCE`. A folder install takes them from a clone or
  an unpacked archive and refuses any file that is not the pinned bytes.

The runtime is a venv without pip, every wheel unpacked; the worker's
`--selftest` runs in the staged interpreter before anything is promoted. An
installed runtime whose closure — wheels, torch's and torchaudio's versions, the
Flash attention wheel and Breeze's commit and files — differs from this build's
is **stale**: Install rebuilds it.

**The model**, `BreezeBlue/Breeze-TTS-2`, is listed rather than named: the hub's
own tree at the commit `main` names, every file with its size and the digest the
hub states (the LFS SHA-256 of a weight, the git blob id of a small file),
fetched at that commit and checked as it lands; the card's README and media are
left out, and a listing without `config.json`, the `audio_tokenizer/` folder
Breeze's loader needs, or a weight is refused before a byte. The first install
records the commit and its files in the overlay, and every later install on that
machine lists the same commit from the record. In case the publisher puts the
model behind its terms, the hub token Voice Chat already keeps (Settings → Voice
Chat → Access token, or `HF_TOKEN`) is offered to huggingface.co and nowhere
else, and a 401 or 403 says to accept the terms and give the token; an open
repository is fetched the same way with or without one.

**Licence.** The inference code is Apache-2.0; the weights are the BreezeBlue
Research and Non-Commercial License, which does not permit commercial use of the
model or of what it renders on your own machine. The install card says so.

---

## 8. Found on the way

* **A cancel could be lost** in both GPU runtimes: the card was marked rendering
  a moment before the render's frame was written, and a cancel in that moment
  reached a worker with nothing to stop, which said no and then ran the whole
  render. A cancel now waits for the frame (found by Breeze's runtime; fixed in
  VibeVoice's too).
* **Whisper's thirty seconds** (section 6), in Voice Chat's dictation as much as
  in the Voice Box.

---

## 9. Not built, and why

* **Takes in one batch.** Breeze's batch endpoint can batch texts through the
  depth decoder but not with CFG; the takes run one after another, which is the
  honest first build.
* **Separate reference and instruction guidance scales**: upstream's command
  line leaves them at none.
* **An upstream test on a tiny model**, as `tests/test_vibevoice_upstream.py` is
  for VibeVoice: Breeze's loader needs a real checkpoint with its codec, and no
  tiny one exists. The worker's calls are checked against stand-ins with
  upstream's exact keywords; the first real render on the owner's machine is the
  proof.
* **Pins this machine could not make**: the model's files, torch's and
  torchaudio's digests and the Flash attention wheel's. `python
  tools/pin_breeze_models.py --model --torch --flash-attn` on a machine that
  reaches the hub and download.pytorch.org turns each into a committed constant.

---

## 10. Risks and open items

* **Nothing of Breeze has run on Windows or on the owner's cards.** Upstream
  tests on Linux. The first things to watch: the install's self-test (Flash
  attention's import above all), the handshake, a one-speaker clone's speed on
  each card, and the peak against the estimate.
* **Speed is unmeasured.** Upstream's figures are an H100 with the fast path
  warmed; eager on a 3090 may be near real time. The fast path is the lever, and
  on Windows it is the CUDA-graph stages only.
* **Dialogue sounds cut together.** VibeVoice renders a conversation in one
  breath; Breeze renders each line alone and the Line gap joins them.
* **A long monologue may show seams** where a line was split.
* **One deadline for a render** (an hour): four long takes of a long section
  could come near it.
* **Two CUDA torches on one machine** — about 10 GB of wheels on disk between
  the two runtimes. That is the point of separate closures.
