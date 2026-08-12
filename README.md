# Model Chain

Cross-architecture two-stage generation for
[Stable Diffusion WebUI Forge Neo](https://github.com/Haoming02/sd-webui-forge-classic)
(the `neo` branch).

Stage 1 completes an ordinary txt2img generation on the loaded checkpoint.
Stage 2 re-encodes the finished **pixels** and runs an img2img refinement pass
on a second checkpoint.

Because the handoff happens in pixel space rather than latent space, each model
uses its own VAE and text encoder. That makes the pairing architecture-agnostic:
SDXL → Flux.2-Klein works, and so does any other A → B combination the WebUI can
load.

> This is **not** a latent-space handoff, and it does not replace or modify the
> built-in Refiner. The Refiner switches the diffusion model partway through a
> single denoising process while staying in one latent space, which restricts it
> to model pairs that share a latent space and conditioning format (SDXL base →
> SDXL refiner, Wan 2.2 high-noise → low-noise). Model Chain is additive and
> solves a different problem.

## Install

Clone into the WebUI's `extensions` directory and restart:

```
git clone https://github.com/RJSprod/SD-Neo-ModelSwitchRefiner extensions/sd-model-chain
```

The panel appears as a **Model Chain** accordion on the txt2img tab.

## Using it

1. Set up your txt2img generation as usual — that is Stage 1.
2. Open **Model Chain**, tick the accordion, and pick a **Stage 2 checkpoint**.
   The detected architecture is shown beside the residency status.
3. If the Stage 2 model needs its own VAE / text encoder files, select them in
   **Stage 2 VAE / Text Encoder**. Flux.2 Klein and Krea 2 do.
4. Set **Denoise strength**. This is the key control: it governs how much
   Model B may alter the Stage 1 image. It defaults to `1.0`, a full Stage 2
   pass; lower it to keep more of the Stage 1 composition.
5. Generate.

The gallery shows only the refined Stage 2 outputs, one per image your batch
settings would normally have produced.

### Presets

**Preset** saves and recalls a complete Stage 2 configuration — checkpoint, VAE
and text encoder, prompt mode and text, styles, seed handling, sampling, size and
edit mode, plus the enable toggle itself.

Type a name, hit **Save**, and it appears in the dropdown. Selecting a preset
applies it immediately; **Delete** removes it. The refresh button re-reads the
file, so presets saved in another tab show up without a restart.

Presets live in `model_chain_presets.json` in your WebUI data directory, not in
the extension folder, so updating or reinstalling the extension keeps them.
Writes go through a temporary file and an atomic replace: a crash mid-save
leaves the previous file intact rather than a truncated one. A preset saved
before a control existed falls back to that control's default rather than
blanking it.

### VAE / text encoder

Flux-family and Krea 2 checkpoints keep their VAE and text encoder in **separate
files**, selected in Forge Neo as "additional modules". The **Stage 2 VAE / Text
Encoder** dropdown selects those independently of Stage 1's:

| Selection | Effect |
| --- | --- |
| **Use same choices** (default) | keep Stage 1's VAE / text encoder |
| specific files | load those for the Stage 2 checkpoint |
| cleared (empty) | use the checkpoint's built-in VAE / text encoder |

This is what makes a genuinely cross-architecture chain work. An SDXL → Flux.2
Klein chain needs Klein's own VAE and Qwen3 text encoder for Stage 2 — inheriting
SDXL's encoder stack is not merely suboptimal, it is the wrong model. The
sentinel and the semantics mirror the host's own **Hires VAE / Text Encoder**
dropdown, so the control should read familiarly.

**Both models keep their own encoders resident.** The host folds the module list
into `forge_loading_parameters`, so a checkpoint paired with a particular VAE and
text encoder is a distinct cache key. Stage 1's pairing and Stage 2's pairing
each occupy their own cache slot, and switching between them is a warm pointer
swap that brings the matching encoders back with the model — no disk read, no
re-encode of the text encoder.

The refresh button next to the Stage 2 checkpoint rescans both lists and keeps
any selection that still exists.

Stage 1's selection is captured before the switch and restored afterwards along
with the checkpoint. Restoring only half of the pair would silently change what
Stage 1 loads on your next generation, and would leave the main VAE/TE dropdown
showing Stage 2's files.

### Prompt modes

| Mode | Stage 2 prompt |
| --- | --- |
| **Inherit** | the Stage 1 prompt, unchanged |
| **Append** | the Stage 1 prompt plus your added text |
| **Replace** | your text only |

Flux-family models respond better to natural-language phrasing than to
comma-separated tag soup, so a Stage 2 prompt often wants different wording than
Stage 1. That is what Replace mode is for.

### Styles

The **Stage 2 styles** dropdown lists your saved styles and applies them to the
Stage 2 prompt in Append and Replace modes. It reads the live style store, so a
style you create mid-session shows up as soon as you hit the refresh button next
to it — no restart. Styles are ignored in Inherit mode, and the dropdown is
greyed out there to make that obvious.

Style expansion uses the WebUI's own `apply_styles_to_prompt` /
`apply_negative_styles_to_prompt`, so `{prompt}` placeholders and multi-style
ordering behave exactly as they do for the main prompt box.

### LoRAs and extra networks

Write `<lora:name:weight>` into the Stage 2 prompt boxes and it applies against
**Model B**, after the switch. Tags inside a saved style work too — styles expand
first, then the expanded prompt goes through the host's extra-networks parser.

The extension never strips, sanitises or pre-parses these tags. The assembled
prompt is handed to the standard processing pipeline, which parses them, applies
them, strips them before the text encoder, and deactivates them afterwards —
byte-for-byte the same path as typing the tag into the main prompt box.

**Stage 1 LoRAs do not carry over.** They are Stage-1-architecture-specific, and
so are embeddings and ControlNet units. When Model Chain detects that the two
stages use different architectures it says so in the panel.

### Edit mode (Krea 2, Anima, Flux.2 Klein)

Some architectures can take the Stage 1 image as an **edit reference** —
vision-conditioning the text encoder and concatenating reference latents — rather
than using it as an img2img starting point. That is how you refine with **Krea 2
Edit**.

The **Stage 2 edit mode** control has three settings:

| Setting | Effect |
| --- | --- |
| **Auto** (default) | follows the model's global Settings toggle |
| **Enable** | forces reference conditioning on, for the Stage 2 pass only |
| **Disable** | forces plain img2img, for the Stage 2 pass only |

To refine with Krea 2 Edit:

1. Pick your Krea 2 checkpoint as the Stage 2 model, and select Krea 2's VAE
   and text encoder in **Stage 2 VAE / Text Encoder**.
2. Set edit mode to **Enable**.
3. Use **Append** or **Replace** prompt mode and add the Krea 2 Edit LoRA:
   `<lora:your_krea2_edit_lora:1.0>`. Krea 2's edit behaviour comes from that
   LoRA — the base checkpoint alone will not do it, and the panel warns if no
   extra-network tag is present.
4. Leave denoise at **1.0** — the default. In edit mode the reference carries
   the content, so a lowered denoise is the wrong control; if you have lowered
   it, ticking Enable raises it back.

The underlying toggles (`krea2_do_reference`, `anima_do_reference`,
`klein_no_reference`) are **global** settings. Enable and Disable are applied
through the host's per-generation `override_settings`, so they are scoped to the
Stage 2 pass and your global value is restored afterwards — the extension never
leaves a global toggle flipped behind your back. Auto touches nothing.

Two details worth knowing:

- **The polarity differs between architectures.** Krea 2 and Anima opt *in*
  (`*_do_reference = True` enables edit mode); Flux.2 Klein opts *out*
  (`klein_no_reference = True` disables it). The control normalises this — Enable
  means "references on" regardless of which way the underlying option runs.
- **Klein is reference-conditioned by default**, and still uses the image as an
  img2img init as well, so an ordinary low-denoise Klein refine is perfectly
  valid and is not warned about. Krea 2 and Anima edit mode is a deliberate mode
  switch away from img2img, so there the denoise and Edit-LoRA warnings do apply.

Each image is refined on its own (batch size 1 per pass), which matters here:
Krea 2 captures only the *first* image of a batch as its reference, so a batched
refine would silently reuse image 1's reference for the whole set.

Stale references are cleared before Stage 2 runs. A cached model keeps its engine
object — and its reference list — across generations, and ImageStitch cannot
clear it because Stage 2 runs with scripts disabled.

### Sampling

**Stage 2 sampling method** and **Stage 2 schedule type** are selected
independently of Stage 1. Both default to **Same as Stage 1**, and either can be
overridden without touching the other — a Flux Stage 2 often wants Euler + Beta
regardless of what an SDXL Stage 1 used.

Whatever you pick goes through the host's own
`fix_p_invalid_sampler_and_scheduler`, so an incompatible pairing is corrected
rather than failing the pass.

### Seeds

Each refine pass reuses the seed from its own Stage 1 image, so
`refined[i].seed == stage1[i].seed`. This is the default and needs no
configuration. **Offset** adds a constant to each inherited seed; **Fixed**
overrides the whole batch with one seed.

### Output size

The size multiplier scales both axes by the same factor, preserving aspect
ratio, then snaps each to the target architecture's required pixel grid — 8 for
SD 1.5 and SDXL, 16 for the Flux, Wan, Qwen and Lumina families. The panel shows
the resulting resolution live as you move the slider. Unrecognised
architectures fall back to 16, which is a multiple of 8 and therefore valid for
an 8-aligned model too.

### Settings

Under **Settings → Model Chain**:

- **Save Stage 1 intermediate images to disk** (default off) — writes the
  unrefined Stage 1 images to a `model-chain-stage1/` subfolder of your output
  directory. They never appear in the gallery.
- **Max system RAM for model cache (GB)** (default 0 = 60% of detected system
  RAM) — the ceiling on RAM held by the residency cache. This is a ceiling, not
  a reservation: the live free-RAM check is the real guard. Raise it if the
  console reports a model being refused.

## How it behaves

### Exactly one checkpoint switch per Generate click

Stage 1 runs to completion for the whole request (batch size × batch count = N
images) with Model A resident throughout. The extension then performs a single
transition to Model B and refines all N images with Model B resident for the
entire loop. Switching per image would dominate the runtime, so it is never
done — the switch count is independent of your batch dimensions, and the console
logs the one transition.

Switching *back* to Model A is deferred rather than done eagerly, which would
cost a second switch. The selection is restored immediately; the model itself is
swapped back at the start of the next generation, as a warm pointer swap.

### Memory residency

The policy is **VRAM-first, demote only under pressure**, with a
VRAM → system RAM → disk cascade that is demand-driven rather than a fixed
evict-on-switch rule. The panel predicts what the next switch will cost before
you generate:

```
Both models fit in VRAM — no offload expected
Insufficient VRAM for both — Model A will offload to system RAM
Insufficient RAM for cache — Model A will reload from disk on switch
```

Roughly: SDXL fp16 is ~7 GB and Flux.2-Klein 9B fp16 is ~18–20 GB, so dual VRAM
residency of that pair realistically wants a 24 GB card. Quantised Flux builds
(GGUF, nf4, fp4, fp8) are substantially smaller and make dual residency workable
on less. The extension handles both cases and never assumes dual residency.

#### How this works on Forge Neo, and one honest limitation

Forge Neo keeps exactly one checkpoint in `sd_models.model_data.sd_model`, and
`forge_model_reload()` unconditionally calls `memory_management.unload_all_models()`
and drops its reference before loading a new checkpoint from disk. Rather than
reimplement offload logic — which the design explicitly rules out — this
extension cooperates with that path:

- `unload_all_models()` calls `ModelPatcher.detach()`, which **moves** weights to
  the patcher's offload device (system RAM). It does not free them. A checkpoint
  normally disappears from RAM only because `forge_model_reload()` drops the last
  Python reference and the garbage collector reclaims it.
- Keeping a checkpoint resident in system RAM is therefore exactly a matter of
  holding a reference to its `sd_model` across the switch. That is what the cache
  in `mc_memory.py` is.
- Restoring a cached checkpoint is a pointer swap: the object goes back into
  `model_data`, `forge_hash` is set to match `forge_loading_parameters`, and the
  next `forge_model_reload()` returns early instead of touching the disk. No
  `unload_all_models()` runs on that path, so VRAM is reclaimed only on demand by
  `memory_management.load_models_gpu()` — which is the demand-driven cascade,
  implemented entirely through host entry points.

The limitation worth stating plainly: **the very first load of a checkpoint must
go through `forge_model_reload()`, and that host function unloads everything
before it loads.** A cold first switch therefore demotes the outgoing model to
RAM even when both would have fit in VRAM together. Every subsequent switch
between two cached checkpoints is a warm pointer swap with demand-driven VRAM
eviction — that is where "both models stay hot in VRAM" and the speedup over a
cold disk load actually come from. Avoiding the first-switch unload would mean
replacing the host's loader, which is a bigger and more fragile change than this
extension should make.

#### Model flags travel with the cached model

`forge_loader` sets a group of `dynamic_args` flags on every load — `kontext`,
`edit`, `nunchaku`, `klein`, `wan`, `pid`, `anima`, `krea2` — that describe
*which model is loaded*. A warm pointer swap never runs the loader, so the cache
snapshots those flags alongside each model and re-applies them (plus
`dynamic_args.reset()`) on reinstatement.

This is not cosmetic. `nunchaku` selects a different LoRA application path,
`kontext` and `edit` decide whether Flux.1 and Qwen-Image use reference
conditioning, `klein` changes sampling, and `pid` changes the latent shape.
Leaving them describing the previously loaded checkpoint would corrupt
generation — and it would show up exactly when chaining two architecturally
different models, which is the whole point of the extension.

The RAM budget check is fail-safe rather than best-effort: host RAM exhaustion
can hang or kill the whole process, so a model that would breach the budget is
released instead of cached, and if system RAM cannot be detected at all the cache
is disabled rather than guessed at.

### Interruption

- **During Stage 1** — aborts before any model switch and returns what Stage 1
  produced, labelled as unrefined.
- **During Stage 2** — returns the images refined so far plus the remaining
  unrefined ones, with a warning naming the split. The batch is never silently
  short.

A failed refine pass falls back to that image's Stage 1 output rather than
dropping it.

### Infotext

Every control is written into the generation infotext under a `Model Chain …`
key and restored when the image is pasted back via PNG Info, Send to txt2img or
drag-and-drop. When the extension is disabled no keys are written at all.

Two things worth knowing about the round trip:

- **The primary infotext describes Stage 1** — Model A, the Stage 1 prompt,
  steps, CFG, sampler, size and seed — with the Stage 2 settings alongside it
  under the `Model Chain` keys. This is deliberate: pasting a refined image back
  into txt2img then reproduces the *whole pipeline*. Recording Model B as the
  main checkpoint would restore a configuration that regenerates B → B, which is
  not what produced the image.
- **The recorded Stage 2 prompt is fully resolved** — post-style-expansion — so
  reproduction does not depend on the style library's current contents. Style
  *names* are recorded separately for readability, and a name that no longer
  exists produces a notice rather than a failed paste.

Values containing commas, colons or newlines are quoted by the host's own
`infotext_utils.quote()` and unquoted on parse, which is what keeps
`<lora:name:weight>` tags intact across the round trip.

### Why a switch is slow, and what to check

Read the console. Every transition is now logged, for example:

```
Model Chain: switched to klein.safetensors (VAE/TE: flux2_vae.safetensors, Qwen3-8B.gguf)
             in 1.2s (warm swap from the RAM cache, no disk read) for 1 image
```

A switch has three possible costs, in increasing order:

1. **Warm swap** — a pointer swap, effectively free. The model is in the RAM
   cache and the weights just move back to VRAM on demand.
2. **VRAM movement** — the weights themselves crossing PCIe. Unavoidable when
   both models cannot sit in VRAM together.
3. **Cold load** — reading the checkpoint from disk *and* the host tearing down
   the outgoing model first. This is the expensive one, typically 20s+.

A cold load on the *first* switch to a model in a session is expected and
unavoidable — the host's loader has to read it once. The console says which case
you are in:

```
Model Chain: first load of klein.safetensors this session; later switches to it
             should be warm swaps from the RAM cache.
```

**If every switch is a cold load**, the cache is refusing the model, and the
console names it:

```
Model Chain: not enough system RAM to cache klein.safetensors
             (14.0 GB needed, 10.6 GB available to the cache)
             — it will reload from disk on every switch.
```

Raise **Settings → Model Chain → Max system RAM for model cache (GB)**. A Flux.2
Klein checkpoint with a Qwen3 text encoder is roughly 14 GB resident, so the
budget has to clear that. The default is 60% of system RAM, and the live
free-RAM check still refuses anything that would push the machine into swap.

**If the switch is warm but still takes ten seconds**, that is weights crossing
PCIe, and the fix is elsewhere:

- The two models have to fit in VRAM *together* to avoid movement entirely.
  SDXL (~7.5 GB) plus Flux.2 Klein 9B with a Qwen3 encoder (~14 GB) is ~21.5 GB,
  which does not leave activation headroom on a 24 GB card — expect movement.
  A more heavily quantised Stage 2 build is what buys dual residency there.
- Launch the WebUI with `--pin-shared-memory`. Pinned host memory roughly
  doubles transfer speed; without it, ~1 GB/s is typical.
- Launch with `--cuda-stream 2` to overlap weight transfer with compute.

Neither flag is something this extension can set for you — they are host launch
options — but both target exactly the "Moving model(s) has taken N seconds"
lines.

The pre-flight status line in the panel predicts which of these you are in for
before you generate.

### Interaction with the built-in Refiner

They cannot run in the same generation. The Refiner restores Model A's UNet
weights in its own `postprocess()` hook, which runs *after* this extension's;
if a checkpoint switch had already happened, those weights would be written into
Model B. When the built-in Refiner is enabled, Model Chain skips itself for that
generation and says so rather than corrupting the loaded model.

## Layout

```
mc_arch.py            architecture detection + per-architecture geometry
mc_memory.py          model residency / cache management
mc_infotext.py        infotext write + paste-field registration
mc_presets.py         named Stage 2 configurations
mc_styles.py          style library integration helpers
scripts/model_chain.py  Script class, UI, orchestration
tests/                pytest suite (runs without a WebUI)
```

Two deviations from the layout in the design document, both forced by how Forge
Neo loads extensions:

- **Helper modules live at the extension root, not in `scripts/`.** `load_scripts()`
  imports *every* `.py` file in an extension's `scripts/` directory as a separate
  script module, so helpers placed there would be loaded twice under two module
  identities — fatal for a module holding cache state. The extension root is what
  gets added to `sys.path`, so root-level helpers import cleanly as top-level
  modules. This is the same structure the bundled `sd_forge_lora` extension uses.
- **`mc_arch.py` and `mc_presets.py` are additional modules**, for architecture
  detection (needed by both the UI and the orchestration code) and for preset
  storage.

`mc_memory.py`'s two-slot cache is private to that module. `ensure_resident()` /
`get_model()` expose no slot count, so swapping in an LRU pool for 3+ models
needs no change to the orchestration code.

## Tests

```
pip install pytest pillow psutil
python -m pytest tests/
```

The suite stubs the host, so it runs without a WebUI checkout or any model
files. It covers the acceptance criteria that can be verified without a GPU:
N-in/N-out batch behaviour, exactly-one-switch sequencing, per-image seed
inheritance, prompt and style resolution, LoRA tag pass-through, sampler and
schedule-type overrides, aspect-ratio preservation and grid alignment, infotext
round-tripping, the residency cascade and RAM budget, per-checkpoint
VAE/text-encoder selection and its cache keying, model-flag restoration across a
warm swap, edit-mode scoping and polarity, preset round-tripping and recovery
from a damaged store, interruption handling, the UI's control-order contract,
and inertness when disabled.

The criteria that need real hardware — that an SDXL → Flux.2-Klein chain
produces coherent output, that a Krea 2 Edit refine responds to its Edit LoRA,
that a LoRA visibly affects the refined image, that a warm switch is measurably
faster than a cold disk load — are left to manual verification.
