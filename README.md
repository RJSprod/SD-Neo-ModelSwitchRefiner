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
3. Set **Denoise strength**. This is the key control: it governs how much
   Model B may alter the Stage 1 image. The default of `0.35` keeps the Stage 1
   composition; higher values give Model B more freedom.
4. Generate.

The gallery shows only the refined Stage 2 outputs, one per image your batch
settings would normally have produced.

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
- **Max system RAM for model cache (GB)** (default 0 = one third of detected
  system RAM) — the ceiling on RAM held by the residency cache.

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
- **`mc_arch.py` is an additional module**, because architecture detection and
  the alignment table are needed by both the UI and the orchestration code.

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
inheritance, prompt and style resolution, LoRA tag pass-through, aspect-ratio
preservation and grid alignment, infotext round-tripping, the residency cascade
and RAM budget, interruption handling, and inertness when disabled.

The criteria that need real hardware — that an SDXL → Flux.2-Klein chain
produces coherent output, that a LoRA visibly affects the refined image, that a
warm switch is measurably faster than a cold disk load — are left to manual
verification.
