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

The panel appears as a **Model Chain** accordion on the txt2img tab, and a
top-level **LLM Studio** tab appears beside txt2img and img2img. The LLM half is
inert until you open it — see [LLM Studio](#llm-studio) — and can be turned off
entirely in Settings.

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

### Hires. fix

Hires fix runs *inside* Stage 1: the first pass is generated, upscaled and
re-sampled, and only that upscaled result reaches Stage 2. Stage 2 then refines
it **once**.

Nothing from the hires pass carries into Stage 2 — not its steps, not its
denoise, not a second upscale. Stage 2 uses its own controls throughout, and the
size multiplier applies to the image it actually receives, so `1.0x` after a 2x
hires pass means "keep the 2x result", not "upscale again".

The panel's size readout follows the hires upscale, so it shows the size Stage 2
will produce rather than the first-pass size. `Model Chain Stage1 Size` in the
infotext records the upscaled size — the image Stage 2 was handed.

That key is *measured*, not predicted. The readout has to work from the width
and height sliders, because it runs while you are still setting up; the recorded
value is read off the finished image. If the two disagree, the infotext is the
one to believe, and the console says so.

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

Tags you write in the Stage 2 boxes are never stripped, sanitised or pre-parsed.
The assembled prompt is handed to the standard processing pipeline, which parses
them, applies them, strips them before the text encoder, and deactivates them
afterwards — byte-for-byte the same path as typing the tag into the main prompt
box.

**Stage 1 LoRAs do not carry over**, and this is enforced rather than assumed.
Inherit and Append modes take their text from the Stage 1 prompt, and that prompt
still holds its extra-network tags — the host strips them from a *copy* on the
way to the text encoder, which is why they survive into infotext. Inherited
verbatim, they would be parsed again by the Stage 2 pass and applied against
Model B. So the inherited half of the prompt has its tags removed before it is
used, and the console says which ones:

```
Model Chain: <lora:sdxl_detail:0.8> in the Stage 1 prompt was not carried into
             Stage 2 — they are Stage-1-architecture-specific. Add a Stage 2 LoRA
             with <lora:name:weight> in Append or Replace mode.
```

This matters most in exactly the case the extension exists for. A LoRA trained
for Stage 1's architecture applied to a different Stage 2 architecture either
fails to load or lands on the wrong tensors, and neither produces an error you
would connect to the prompt. Embeddings and ControlNet units are
architecture-specific in the same way; when Model Chain detects that the two
stages differ it says so in the panel.

To use a LoRA during refinement, put it in the **Stage 2** prompt box, in Append
or Replace mode.

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

### Stage 2 reference images

Edit mode above is about the **Stage 1 image**. This control is about
*supplemental* references — extra images every Stage 2 refinement sees, on top
of its own Stage 1 handoff.

Forge's **ImageStitch Integrated** does the same job for Stage 1. Despite the
name it does not stitch anything: it hands the model an ordered set of reference
images. Model Chain runs Stage 2 through its own processing object with scripts
disabled, so ImageStitch never runs for the refine pass — which is why Model
Chain routes Stage 2's references itself rather than trying to re-run
ImageStitch there.

| Mode | Effect |
| --- | --- |
| **Disabled** (default) | Stage 2 gets no supplemental references. Stage 1 ImageStitch is untouched. |
| **Pass Through ImageStitch** | The images in ImageStitch's gallery are reused for every Stage 2 image, in the order shown there. |
| **Decoupled** | Model Chain's own gallery is used instead, and Stage 1's ImageStitch references are not included. |

In both active modes the references are **shared across the batch** and each
Stage 2 image still keeps its own Stage 1 result as the primary reference:

```
Stage1-A + Ref1/Ref2/… -> Stage2-A
Stage1-B + Ref1/Ref2/… -> Stage2-B
```

Order matters and is never rearranged. The Decoupled gallery has the same
append / replace / delete / clear controls as ImageStitch's, so a set can be
built and reordered without leaving the panel.

Pass Through only ever *reads* ImageStitch's gallery, and uses ImageStitch's own
**Maximum Side Length** for sizing. Your Stage 1 selection is never cleared,
reordered or replaced. Decoupled has its own maximum-side-length slider, since
Model Chain owns that set.

**Which models can use them.** Reference conditioning is an engine feature, not
a universal one:

| Stage 2 architecture | Support |
| --- | --- |
| Flux.2 Klein (4B / 9B), Krea 2 | validated |
| Anima, Qwen-Image-Edit, Flux.1 Kontext | experimental — Forge exposes the path, this pairing is not validated here |
| everything else | no reference path; references are ignored with a notice |

The mode stays selectable whatever you pick, and the panel says what will
happen. Nothing is decided on the checkpoint's name, and nothing is *refused* on
a guess made before the model loads:

- The panel's verdict comes from reading the checkpoint header, which is the
  best that can be done before anything is loaded. It is a prediction.
- Once Stage 2's model is in memory it is asked directly — Forge hands each
  engine the config class its own detector settled on, so this is the loader's
  own answer rather than a second opinion. A checkpoint the header could not
  identify (a quantised or repacked build, or a GGUF, which cannot be read at
  all) is settled here instead of being turned away.
- For Flux.1 and Qwen-Image even the architecture is not enough — only the
  Kontext and Edit variants have the path — so the loaded model's own flag is
  checked too.
- After encoding, the number the model actually accepted is compared against
  the number supplied. If they differ the whole set is dropped, because a
  partial set no longer carries the order you arranged.

A checkpoint that genuinely has no reference path has its references dropped
with a notice, never silently mis-fed.

**Interaction with edit mode.** Supplying references means asking for reference
conditioning, so choosing Pass Through or Decoupled implies **Enable** for the
Stage 2 pass. Setting edit mode explicitly to **Disable** overrules that — the
references are dropped and the panel and console say so, rather than one control
silently reversing another.

**Reference images are not stored in PNG info**, exactly as native ImageStitch
does not store them. The infotext records the mode (and the reference count, for
diagnostics). Pasting an image restores the mode; you re-add the images
yourself, and generating without them produces a notice rather than a failure.
Presets follow the same rule: they save the mode and the maximum side length,
never the pixels.

References are cleared at both ends of every job — before the references are
added and after the last pass, on success and on failure alike — so a cached
Stage 2 model can never carry a set into the next generation. Because clearing
them can outdate ImageStitch's own memo of what it last encoded, that memo is
reset too, so Stage 1 re-encodes its references next time instead of assuming
they are still there.

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

The multiplier scales **the image Stage 2 was handed**, not the width and height
sliders. Those two are usually the same number, but not always: hires fix
changes it, and so does anything that adjusts the pass dimensions before
sampling. Every generation logs the whole chain of sizes at INFO, so when the
output is not the size you expected the console tells you where it changed:

```
Model Chain: Stage 2 refines 960x1280 at 1.00x — requesting 960x1280, aligned to 16px for Flux.2 Klein 9B
```

If Stage 1 produced something other than the sliders asked for, a line above
that one names both sizes. And if the refine pass returns a size other than the
one it was asked for, that is reported as a warning and added to the image's
comments rather than shipped silently — the extension chooses what to
*request*, but the pass decides what comes back.

### Settings

Under **Settings → Model Chain**:

- **Save Stage 1 intermediate images to disk** (default off) — writes the
  unrefined Stage 1 images to a `model-chain-stage1/` subfolder of your output
  directory. They never appear in the gallery.
- **Max system RAM for model cache (GB)** (default 0 = 60% of detected system
  RAM) — the ceiling on RAM held by the residency cache. This is a ceiling, not
  a reservation: the live free-RAM check is the real guard. Raise it if the
  console reports a model being refused.
- **Preload Stage 1 after Stage 2 finishes** (experimental, default **off**) —
  see [Preloading Stage 1](#preloading-stage-1-experimental-off-by-default).
- **Keep Stage 1's text encoder and VAE in VRAM during Stage 2** (default on) —
  see [Pinning Stage 1's encoders](#pinning-stage-1s-encoders).
- **Use leftover VRAM to keep Stage 2 warm between generations** (default on,
  but only takes effect when the preload above is enabled) — see
  [Spending what is left](#spending-what-is-left).
- **Minimum VRAM reserve (GB)** (default 0 = automatic) — a floor under the
  automatically sized reserve. Model Chain never warms anything into it, and
  never undercuts VRAM Forge itself has set aside. See
  [The reserve](#the-reserve).
- **Reuse a prepared LoRA state when a cached model comes back** (default on) —
  see [Prepared LoRA state](#prepared-lora-state). Turn it off if a LoRA
  misbehaves after a switch; every restored model then rebuilds its LoRA state
  from scratch, which is slower but leaves nothing to be wrong about.
- **Predict progress and ETA for the whole chained job** (default on) — see
  [Progress and ETA](#progress-and-eta).
- **Custom progress-bar appearance**, **Progress-bar theme**, **Progress-bar
  colour** and the toggles below them — see
  [Progress-bar appearance](#progress-bar-appearance).

### Progress and ETA

The WebUI's progress bar counts sampler jobs and steps. That describes one
sampling loop well and a chained job badly: the bar would reach 100% when
Stage 1 finished and then drop back when Stage 2's images were added to the
count. A batch of two, 20 steps then 8, went 100% → 33% → 66% → 100%. Time
spent moving several gigabytes of weights between the stages counted for
nothing at all.

With this on, the job is modelled as timed phases — waiting on a preload,
freeing VRAM, Stage 1 sampling, the checkpoint switch, Stage 2 sampling,
finalising — and the bar reports **how much of the predicted wall time has
passed**. It moves forwards only, at a roughly even rate through work of very
different kinds, and the ETA falls out of the same arithmetic. The bar's text
names the phase, so `Stage 2 1/2` appears beside the percentage without the
percentage jumping for it.

The predictions are **measured on your machine**. The first chained generation
on a fresh install works from a small built-in table, scaled by your card's
VRAM; from the second onwards it uses what it actually observed — seconds per
gigabyte for each kind of model switch, seconds per step per megapixel for each
architecture and batch size. Those measurements are written to
`model_chain_timing.json` in your WebUI data directory, so they survive a
restart. Interrupted and failed jobs are not learned from.

Batch size is part of the model, and asymmetrically so: Stage 1 samples a batch
in one pass, but Stage 2 refines each image separately, so it gets none of the
batching gain. Hires fix is accounted for too, including that its two passes
cost different amounts.

The bar never shows 100% — the WebUI removes it when the job genuinely ends. And
"the job" means your images are ready. If the optional
[Stage 1 preload](#preloading-stage-1-experimental-off-by-default) is on it runs
in the background *after* that, once the progress bar is already gone; the
panel's residency line is where its state is reported.

Nothing here is required. With it off, the bar falls back to the WebUI's own
calculation — still without the backwards jump, because the job counters are
sized for the whole chain before Stage 1 starts either way.

### Progress-bar appearance

Purely cosmetic, entirely independent of everything above, and applied to
**every** generation. It never changes the numbers on the bar; the calculation
above never changes how it looks. Either can be turned off without affecting
the other.

### Smooth advance

**On by default, and the one setting here that needs nothing switched on.**

The WebUI writes the bar's width once per progress poll — every *Progress bar
update period* milliseconds, half a second out of the box — so the bar arrives
in visible steps. This fills in the movement between those writes.

The numbers the WebUI reports are untouched, and so is the width it writes —
only what is drawn between two of its own values changes. It applies to every
generation on every tab, with or without Model Chain, with or without a theme
selected below.

The bar is redrawn **every frame**, with a speed of its own. It tracks how fast
progress is really moving, keeps advancing at that speed between polls, and
steers gently towards wherever the next value is predicted to land — so a change
in the real rate arrives as a change in *speed* rather than as a jump. It never
stops, never steps, and never snaps to a reported number.

A plain CSS transition cannot do this, which is worth knowing if you were
expecting one: each new value retargets the transition, so the bar's speed
changes abruptly at every poll. That is finer-grained stepping, not continuous
motion. A transition is still used as a fallback if the script cannot attach.

Measured over a simulated generation, sampling the rendered width every frame:

| | Off | On |
| --- | --- | --- |
| Frames in which the bar moved | 2% | **100%** |
| Frames in which it sat still | 175 of 179 | **0** |
| Fastest frame | 1912 px/s | 47 px/s |

### Themes

**Installing the extension is all it takes.** The theme applies whether or not
Model Chain is enabled, whether or not a Stage 2 checkpoint is selected, and
whether or not you have ever opened the accordion — and on **txt2img, img2img
and Extras alike**, even though Model Chain itself is txt2img-only. It restyles
the WebUI's own progress bar, so there is no version of the bar that belongs to
this extension and no state of the extension that can switch it off. The only
thing that turns it off is its own toggle, or uninstalling.

Tick **Custom progress-bar appearance**, then pick a **Progress-bar theme**:

| Theme | Look |
| --- | --- |
| **Flat** (default) | plain fill — what the WebUI does, in your colour |
| **Gradient** | the fill lightens towards its leading edge |
| **Sheen** | the above, plus a highlight band travelling along the bar |
| **Pulse** | flat fill with a breathing halo around it |
| **Neon** | gradient, sheen and a hard glow — the loud one |
| **Ooze** | glowing sludge that fills from the left, bubbling, with bubbles rising out of the surface and popping above the bar |
| **Custom** | build your own from the three toggles below |

**Progress-bar colour** applies to *whichever* theme you picked — every theme
derives its gradient stops and glow from one colour, so setting it recolours the
whole look rather than only the plain one. Any CSS colour works: `#38bdf8`,
`rgba(56, 189, 248, 0.85)`, `hsl(199 89% 60%)`. Leave it empty to follow your
WebUI theme's own accent colour, which is what keeps it looking right in light
and dark. Something the browser cannot parse is ignored, with a note in the
console, rather than blanking the bar.

Try `#39ff5e` with **Ooze** for toxic slime, `#b026ff` for something more
radioactive, `#ff6a00` for molten.

The three **Custom theme** toggles — fade, sheen, glow — do nothing unless the
theme above is set to `Custom`.

**Ooze is the one theme that takes the bar over completely**, and it is the only
one that overrides what a WebUI theme does there. Two things: it re-enables
overflow, because Lobe hides anything leaving the bar and the bubbles have to
leave it; and it switches off Lobe's animated diagonal stripes across the fill,
which over sludge read as candy rather than as a surface. Lobe's softer top
highlight is left alone. Every other theme here layers with whatever your WebUI
theme draws — Ooze is the exception, because half a takeover looks worse than
either whole.

The rising bubbles are real elements rather than a background pattern, so each
has its own size, speed and drift. They are parented to the bar's *track*,
which the WebUI builds once, rather than to the fill, whose contents it rewrites
twice a second, and they are added by a `MutationObserver` watching for the bar
to appear — which runs after the fact and so cannot interfere with the
generation itself.

If your system asks for reduced motion, most of these themes stop moving
entirely and the completion flash does not fire. **Ooze slows down instead of
stopping**, because motion is the whole content of that theme rather than
decoration on top of it — a still Ooze is a flat green bar under a field of
dots that never move, which looks broken rather than considerate. Everything
runs about three times slower, the sideways drift goes, and the bubbles fade out
rather than bursting.

**Flash the bar when a job finishes** works with every theme. It fires once, at
the end of the whole job; in a chained generation that is after Stage 2, and it
cannot fire at the end of Stage 1, because both stages run inside one
progress-bar lifecycle.

Everything animated respects your system's reduced-motion setting, and the glow
is drawn *outside* the bar so it can never make the percentage and ETA harder to
read.

#### Living with other WebUI themes

The appearance layer is built to share the bar rather than fight for it. It sets
no geometry — no height, offset or radius — so the bar stays wherever your theme
puts it, and it uses no `::before`/`::after`, so a theme that already decorates
the bar keeps its decoration.

[Lobe Theme](https://github.com/lobehub/sd-webui-lobe-theme) was the case
checked in detail. It restyles the WebUI's own bar rather than replacing it, it
never sets the fill colour, and it redefines the same `--primary-*` variables the
default here reads — so with **Flat** and no colour set, the bar picks up Lobe's
accent instead of the WebUI's hardcoded blue, which is an improvement on the
current appearance rather than a change to it.

Lobe also animates diagonal stripes over the bar. **Sheen** and **Neon** would
otherwise put a second moving pattern on the same small element, so the sheen
detects that and stands down by itself, leaving the colour and glow to carry the
look. That check measures the actual conflict rather than looking for a theme by
name, so it works for any theme that decorates the bar the same way.

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

#### So does the latent scale

`forge_model_reload()` ends by writing `modules.processing.opt_f` from the VAE
it has just loaded — 16 for Flux.2, 8 for everything else, including a Wan VAE
whose ratio is a tuple rather than an int. That single module-level number is
what `process_images_inner` divides the requested pixel size by to shape the
noise, so it has to describe the model that is about to sample.

A warm swap runs no loader, so the extension writes it too, on both directions
of the switch. Without that, a Krea 2 → Flux.2 Klein chain is correct on the
first Generate — both models are loaded from disk — and wrong on every one
after it: Krea 2 comes back from the RAM cache under Flux.2's 16, a 640×960
request becomes a 60×40 latent, and Krea 2's VAE decodes it to a 320×480 image.
Stage 2 then refines exactly what it was handed, at half size, which is where
the user sees it and where the extension gets the blame.

Because the value is a global that any code path can leave stale, it is also
checked at the top of every generation and corrected — loudly, since arriving
there means a swap happened that nothing accounted for — while correcting it is
still free. The geometry line in a wrong-size report names both the divisor in
force and the one the loaded model's VAE asks for; those disagreeing *is* the
wrong size.

The RAM budget check is fail-safe rather than best-effort: host RAM exhaustion
can hang or kill the whole process, so a model that would breach the budget is
released instead of cached, and if system RAM cannot be detected at all the cache
is disabled rather than guessed at.

The budget is not a fixed ceiling either. It is the lower of the configured
maximum and *what the cache already holds plus the RAM actually free*, so it
tracks the machine rather than a number chosen once — which is why the same
pairing can hold both models on one generation and only one on the next.

#### Which entry gets evicted

Recency alone picks the wrong victim at exactly the moment a switch happens.
Every switch stashes the outgoing model moments before restoring the incoming
one, and the incoming model has not been touched since the previous generation
— so it is the least recently used, so plain LRU throws out the very model that
is about to be installed. It then reloads from disk immediately.

Both switch points therefore name the entry they are about to fetch, and the
cache will not evict it. When the budget only fits one model, the model being
*put away* is refused instead: that costs a disk load on some later switch,
where evicting the one being *fetched* costs one now, and the deferred cost may
never be paid at all if free RAM recovers first. The refusal message says which
of the two happened, because "not enough system RAM" on its own reads as a
fault when the cache has in fact just made the right choice deliberately.

#### Pinning Stage 1's encoders

When two models cannot both fit, the switch is dominated by moving weights, and
the encoders are a disproportionate share of it. One measured Krea 2 → Flux.2
cycle spent 3.7s moving Stage 1's text encoder against 5.6s for its UNet — for a
component a fraction of the size.

So when making room for Stage 2, Stage 1's text encoder and VAE are spared and
only the UNets are swapped. Stage 2's own encoders stay evictable, which is the
asymmetry that makes it worthwhile: Stage 1 is the model you always come back to.

Two conditions have to hold, and both fail quietly and visibly in the log:

- **Stage 2 must still fit alongside them.** Pinning more than the card can
  spare is worse than not pinning at all — the eviction would fall short of its
  target and the host would then make up the difference by partially unloading
  *while* it loads, which is the slow path this is trying to avoid.
- **The switch must be a warm one.** A cold first load of Stage 2 goes through
  `forge_model_reload()`, which calls `unload_all_models()` and takes everything
  down regardless. From the second generation onward the switch is warm and the
  pin applies.

#### Prepared LoRA state

Forge's LoRA loader does not patch weights on every generation. It builds a
patched clone of `forge_objects` once and records what produced it in
`sd_model.current_lora_hash` — a string built from the network names, their
text-encoder and UNet multipliers and any dynamic dimensions. The next call
compares that string and returns early when it matches.

Because this extension caches whole `sd_model` objects rather than reloading
them, that hash travels with the cached model, and the early return keeps
working across a warm swap. **So there is nothing here that reapplies a LoRA, and
nothing that moves patched state around by hand.** Preserving the prepared state
is a matter of not breaking the host's own mechanism, and the work is entirely in
knowing when preserving it would be wrong:

- **The other stage never inherits it.** Each entry records which stage prepared
  it. Two different checkpoints are two different cache entries and the question
  never arises, but one checkpoint used by both stages carries a single prepared
  state belonging to whichever stage ran last — and handing that to the other one
  is the same cross-stage leak the prompt stripping prevents.
- **A state that changed while the model sat in the cache is not trusted.** The
  recorded hash and the live one are compared on the way back in.
- **Backends that rebuild rather than move are excluded.** Nunchaku folds LoRA
  weights into a quantised kernel instead of a patcher clone, so its prepared
  state is not the movable object the rest of this assumes. Those models fall
  back to the host's normal path and the console says so.
- **A failed refine invalidates it.** A pass that raised part way through LoRA
  application leaves the host believing a state is applied that partly is not.
  In a stock session that belief dies with the next checkpoint load; here the
  model object is kept, so the belief would outlive the job that created it.
  Throwing it away costs one reapplication.

Changing the LoRA name, its weight, the on-the-fly patching mode or the model
needs none of this: those all change the host's own hash, so the host reapplies
without being asked to. What is above is only about the cases where the hash
would *match* and should not be believed.

Invalidation is always the conservative direction — the worst it can do is make
the host redo work it could have skipped. Nothing here ever writes a hash, only
clears one.

#### Preloading Stage 1 (experimental, off by default)

Restoring Stage 1 is two costs. The pointer swap out of the RAM cache is cheap;
moving the weights back into VRAM is not, and it happens lazily the moment the
sampler first asks for them. Left to the next Generate click, you wait through
the whole move before a single step runs.

The preload starts as soon as Stage 2 finishes, on a background thread, so it
overlaps the time you spend looking at the images you just got. Running it
inline would gain nothing — `postprocess` runs before the generation call
returns, so an inline preload would just move the same wait to before the
gallery appears.

**"Preloaded" means ready to sample, not "a thread ran".** The worker swaps
Stage 1 back in, budgets VRAM for it, moves its weights through the host's own
`load_models_gpu`, and then *measures* how much of the model the host reports
resident. A preload that ran out of room reports partial rather than success,
and every generation opens with a line saying which it got:

```
Model Chain: Stage 1 is warm — 12.4 GB already in VRAM (preloaded in 8.5s).
Model Chain: Stage 1 is partially warm — 7.2 GB of 12.4 GB in VRAM, the rest
             moves on demand (preloaded in 5.1s).
Model Chain: Stage 1 is cold — 12.4 GB still to move from system RAM (preload off).
```

That line is measured against the host's live view every time, not reported from
what the preload believed it achieved.

##### What happens when it goes wrong

Nothing downstream depends on it completing, and the failure behaviour is the
part worth trusting:

- **A Generate that arrives mid-preload waits, and the wait is never wasted.**
  The next generation joins the thread before touching any model and then takes
  the same lock, so the worst case is exactly the wait you would have had anyway.
  A wait long enough to notice is logged, so a slow first click has a visible
  explanation rather than looking like a hang.
- **A single failure costs one generation, not the session.** If the swap never
  happened, nothing moved. If it did, Stage 1 *is* the loaded model with its
  weights in RAM — which is precisely the state an ordinary generation starts
  from, so the host's synchronous path takes over with nothing to undo. The next
  generation still budgets VRAM for it, which is the difference between the
  normal load and the pathological one that squeezes a UNet into 900 MB of spare.
- **Nothing survives that might no longer be true.** A failure drops the pinned
  encoders and invalidates the prepared LoRA state rather than trusting either.
- **Repeated failure retires the feature.** Two consecutive failures take the
  preload out of service for the rest of the session, with a line in the console
  saying so. Enabling it can cost you a generation; it cannot cost you every
  generation. Toggling the setting off and on gives it another chance without a
  restart.
- **A checkpoint change supersedes it.** The preload records what it warmed, and
  a change of checkpoint, VAE, text encoder or storage dtype since then means the
  result is discarded and the current selection loads normally.

**It still ships off, and that is deliberate.** See the reasoning below — the
failure handling above changes what a bad outcome costs, not how likely it is.

##### Why it is still off by default

It is the only part of this extension
that touches models away from the generation thread, and that turns out to be
sharper than it looks. `torch.inference_mode()` is thread-local and the host
holds it for a whole generation, so every model the host loads yields *inference
tensors*. A thread starting with grad enabled produces the other kind, and
mixing them is not a slow path but a hard failure — the first sampling step
raises `RuntimeError: Inference tensors do not track version counter`. The
worker now enters the same context the host loads under, which should close
that, but "should" is doing real work in that sentence: the failure depends on
the host build, on other extensions wrapping the sampler, and especially on
**Patch LoRAs on-the-fly**, which makes every load rewrite weights rather than
just move them.

The asymmetry is what decides the default. Pinning and the VRAM sizing *remove*
work; the preload only moves the same work earlier. A machine where it
misbehaves loses far more than a machine where it is off gains, and the failure
handling above changes what a bad outcome costs without changing how likely one
is. Turn it on in **Settings → Model Chain** once you have confirmed it is safe
on yours — and if you are chasing a problem, turn off on-the-fly LoRA patching
first.

#### Sizing the VRAM budget

Two numbers matter, and they are not the same: what the pass **needs**, and how
much has to be **freed** to get there.

The requirement comes from the model's own patchers whenever it is the loaded
model — which, at both points where room is made, it is. That is its true
resident size. Only for a model that is not loaded does the budget fall back to
the file size plus a proportional overhead, because a file is a poor proxy in
both directions: quantised formats read larger than they land, mixed-precision
builds land larger than they read. Under-counting is what produces
`Unloaded partially` immediately before a UNet load — the eviction hit its
target and stopped, the target was short, and the host made up the difference
the expensive way.

What has to be freed is the requirement **minus whatever the target already
holds**, and the target is never itself offered up for eviction. Skipping that
subtraction is self-defeating: after a preload the target is the largest
evictable thing on the card, so asking to free the whole requirement evicts
exactly the weights the next pass is about to use, and the load happens twice.
That was a real bug — the preload's 8.5s of work was being discarded and redone
on every Generate click.

#### The reserve

Every eviction decision is a subtraction from free VRAM, and the thing being
subtracted is the reserve: what has to stay free for the pass to run without the
driver quietly spilling into system memory. Four figures answer that question,
and **the largest of them wins**:

| Floor | Where it comes from |
| --- | --- |
| Static estimate | 1 GB plus 0.75 GB per megapixel above the first |
| Observed peak | the largest activation peak measured this session, plus 15% |
| Manual reserve | your **Minimum VRAM reserve (GB)** setting, if you set one |
| Host reservation | whatever Forge itself reports set aside for inference |

Taking the maximum rather than a sum is the point: each is an answer to the same
question, so the strongest answer is the useful one and none of them may be
undercut by the others. In particular a manual reserve is a *floor*, not an
override — setting 2 GB does not permit a 4096×4096 pass to run on 2 GB — and
Forge's own reservation is never cancelled or reduced by this extension.

**Automatic mode learns, within bounds.** The static estimate cannot know your
sampler, your batch size, or what else is hooked into the UNet, so after each
pass the peak allocation is compared against the weights resident at the time
and the difference is folded in as bytes-per-megapixel. The figure only ever
moves upward: a pass that happened to be cheap is not evidence that the next one
will be, and an under-reserve is the failure with no error message attached.

The bounds matter as much as the learning, because the measurement is biased
high by construction — anything that left the card during the window is counted
as activations. So the learned rate is capped at **twice the a-priori 0.75 GB
per megapixel**, and the reserve it produces at **a quarter of the card**. Both
caps are deliberately expressed in units that do not depend on the pass that
produced the reading. An earlier version capped the rate at a multiple of the
static estimate instead, and because that estimate is mostly a flat 1 GB,
dividing it by a small pass's megapixels let a 512×512 observation authorise
15 GB per megapixel — which the session then applied to everything larger. One
log reached 7.4 GB/megapixel and reserved 10.4 GB for a 1280×960 pass: more
than the 13.9 GB model it was protecting.

**And the requirement never exceeds the card.** A target larger than physical
VRAM is not demanding, it is impossible, and it fails in the worst available
way: `free_memory` evicts everything it is allowed to, still reports a
shortfall, and the pass runs having thrown away models it could have kept. When
model plus reserve would not fit, the *reserve* is what gives — the model has to
be resident to sample at all, whereas a margin that cannot be honoured is better
spent than pretended.

#### Spending what is left

Stage 1 comes first and always. But a 24 GB card running a pair of quantised
models is routinely left with several GB doing nothing once Stage 1 is back, and
that capacity has an obvious use: Stage 2's components, which the *following*
generation wants and would otherwise drag back across PCIe from scratch.

So once Stage 1 is warm, anything still spare beyond the reserve is spent on
Stage 2:

```
Model Chain: kept 2 of Stage 2's 3 components warm in VRAM (3.0 GB moved,
             1.4 GB free, 1.0 GB reserved) — a Stage 2 retry starts sooner
```

**This runs on the preload's thread and nowhere else, so it needs the preload
enabled.** That boundary was learned the hard way and is worth stating plainly.
Model Chain frees VRAM from anywhere — `free_memory` moves weights *out*, which
is what the host does on every eviction anyway — but it fills VRAM from exactly
one place. Loading weights *in* rewrites and re-patches them, and doing that
from a script hook rather than from inside the sampler left the model in a state
the very next sampling step rejected outright with `RuntimeError: Inference
tensors do not track version counter`. The preload has a deliberate answer to
that, an opt-in switch and a circuit breaker; a hook on the generation thread
has none of the three. With the preload off, nothing is captured and nothing is
warmed.

Four rules keep the rest from becoming the problem it is trying to solve:

- **Free VRAM is not the budget.** Free VRAM *minus the reserve* is, and there
  is a further half-gigabyte of slack on top — speculative work should only
  happen where there is room to be wrong about it. On a card that is tight for
  the pair, nothing is warmed and the console says why.
- **Components are dropped largest-first.** A UNet that does not fit must not
  mean nothing is kept: Stage 2's text encoder and VAE are small and a
  disproportionate share of a switch's cost, so they are what remains when the
  UNet is dropped.
- **Warm components are the first thing released under pressure.** They are in
  no keep-list, so the next pass that needs room takes their VRAM back through
  the ordinary eviction path without anything special happening.
- **Nothing is held when nothing can use it.** The captured components are
  references to real weights, so with the preload off they are not captured at
  all rather than keeping gigabytes alive for a warm-up that will never run.

One limitation worth stating: whether keeping a given component warm is
genuinely faster than releasing and reloading it is decided here by whether it
fits, not by measurement. Measuring it properly needs per-component movement
timings that this extension does not yet collect, so the conservative rule — only
ever use capacity that is provably spare — stands in for the measurement.

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

**Both stages get a clean VRAM budget.** When the two models cannot sit in
VRAM together, whichever one is about to run is given room first — Stage 2
before its refine loop, Stage 1 after its model is swapped back in. Without
that, the incoming model loads into whatever the other one left behind and the
host makes room the slow way, partially unloading in small chunks while it
loads. Measured on a Krea 2 → Flux.2 chain, the same ~8 GB UNet took **11.5s**
squeezed into 900 MB of spare VRAM against **0.8s** with 7 GB spare.

When both models genuinely fit, nothing is evicted and both stay hot.

**If sampling itself crawls** — tens of seconds per step where it was
sub-second — VRAM is over-committed and the driver is spilling into system
memory. On Windows that happens silently: no error, just a 50x slowdown. The
console shows it as `0.00 MB usable` on a load, or `lowvram patches: N` on a
model that should have fit. Model Chain evicts the other model from VRAM before
Stage 2 when the pass will not fit alongside it, and warns when even that is not
enough:

```
Model Chain: still only 2.1 GB VRAM free against 17.2 GB needed for a
             2048x2048 pass. Expect the driver to spill into system memory...
```

Hires. fix makes this much more likely, because Stage 2 then refines at the
upscaled size — a 2x hires pass quadruples the pixels Stage 2 has to work on,
and attention activations grow with them. A smaller Stage 2 size multiplier, a
lower hires upscale, or a more heavily quantised Stage 2 model each reduce the
pressure.

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

## LLM Studio

A local-LLM workspace, in the same extension and on the same card. It is the
[LTX_Video_Prompt_Claude](https://github.com/RJSprod/LTX_Video_Prompt_Claude)
application's feature set brought into Forge as a native Gradio tab rather than
as an embedded Qt window: the prompt engine, the conversation system, the
MiniMax enhancer and the llama.cpp runtime are the same code, vendored under
`prompt_master/` with its provenance recorded in
`prompt_master/VENDORED_FROM.txt`. The presentation layer is new; nothing
underneath it is.

It lives here rather than in a separate extension for one reason: **whoever
decides when a model leaves VRAM has to decide it for both kinds of model.** Two
extensions independently unloading things from the same GPU is how you get a
checkpoint evicted for an LLM that then evicts itself for the checkpoint. Model
Chain already owns image residency, so it owns the LLM's too.

### Four modes

| Mode | What it is |
| --- | --- |
| **Prompt Studio** | LTX video-prompt generation. Text-to-video and image-to-video, a positive and a negative prompt as two separate outputs, the smart-negative second pass, and the full control set — style, motion, camera, transition, POV, wardrobe, accent and strength, dialogue budget, extra speech, music, duration, FPS, dimensions, output format, lexicon, extra negative terms and seed. |
| **Conversation** | Threaded chat with persistent histories, and per-message actions. Characters are files in a `characters/` folder in the layout oobabooga uses, so cards import and export; chats are documents filed per character. |
| **MiniMax H3** | The prompt enhancer, as its own workflow. FL2VA and REF2VA variants, an optional reference frame that is captioned first and shown to you, and its own history. |
| **Setup** | The runtime, which GGUF runs, what context fits, and what is currently resident on the card. Everything here needs a file dialog, a download, an estimate or a table — the plain values live on the Settings page instead. |

They share one loaded model and one runtime. They do not share a history, an
output format, or a screen. Switching modes hides a workspace rather than
rebuilding it, so a half-read reply survives a trip to Prompt Studio and back.
The tab opens on the mode you left it on.

### Switching models

The top of the tab is one line: the mode selector, a model **chooser**, a
rescan, **Load**, **Unload**, and a state chip reading *Loaded*, *Unloaded*,
*Loading…* or *Not set up*. Hovering the chip gives the detail — which model, on
which device, at what context — and Setup\u2019s residency view has all of it in
full. It is deliberately small: everything the top bar takes is taken from the
conversation under it.

The chooser lists every `.gguf` under your models folder — the **LLM models
folder** setting, or `models/` inside the LLM data directory — walked a few
levels deep so a `publisher/repo/model.gguf` layout is found, with vision
projectors and every shard of a split model but the first left out, because
neither is something you would choose to *run*. Whatever is recorded right now
is always in the list too, even when it lives somewhere else entirely.

Choosing records the model; it does not load it. **Load** starts llama-server
on what is recorded and reports where it landed — the placement, the context it
got, and anything that had to be reduced to fit. **Unload** stops it and gives
back every byte of VRAM. Both are also what happens on their own when you send
a message or when an image generation needs the card, so the buttons are for
when you want to decide rather than for making it work.

Switching model does not carry the vision projector across, and does not guess
one from the new model's folder. A projector has to match the model it was made
for and a filename does not prove that it does; one sitting beside the new model
is mentioned, and applying it is a press in **Setup**.

### Conversation, per message

Click a message in the transcript and the actions for that message appear under
it:

| Action | What it does |
| --- | --- |
| **Edit** | Rewrite the message in place. The version showing is the one changed. |
| **Regenerate** | Ask for the reply again, keeping the one it had. `◀ 2/3 ▶` pages between attempts, so one that came back worse is undone rather than re-rolled. |
| **Continue** | Carry the last reply on from exactly where it stopped. |
| **Send again from here** | Answer one of your own messages again, dropping everything after it. |
| **Branch from here** | Copy the thread up to this message into a new one. The thread it came from is untouched. |
| **Delete message** / **Delete from here** | One message, or that one and everything after it. |

Tapping the same message again puts the bar away; tapping a different one moves
it there. There is no per-message copy button — select the text and copy it.

The transcript follows a reply while you are at the end of it and holds your
place while you are not: scroll up to read something and new messages arrive
below without moving what you are looking at; scroll back to the bottom and it
starts following again.

The threads list, the character (chat with, edit, or create) and your persona
are in a drawer behind **☰ Threads & character**, closed by default — with it
closed the conversation has the whole browser window. The workspace sizes itself
to the window: the composer is measured first and sits on the bottom edge, the
transcript takes whatever is left, and the page does not scroll — the thread
does.

### While it is working

A two-pixel bar sweeps along the bottom of the status line whenever a request is
in flight, with the seconds it has been running beside it, and both go away when
the run ends. It sits inside the status line that is already there, so nothing
on the page moves when a reply starts. It is indeterminate on purpose — nothing
knows how long a reply will be — and under `prefers-reduced-motion` it stops
moving and stays lit. In the top bar, the state chip pulses while the model is
loading.

### What the console says

Every run reports itself to the WebUI console: llama-server starting with the
model, device, placement and context; a run starting, what it is waiting for,
what it is doing, and a progress line every few seconds while it generates; how
it ended and how long it took; and llama-server stopping with the VRAM released.
Load, Unload and a model change are logged too.

None of it is content. No prompt, no reply, no message, no character name, no
thread title — only what kind of run it was, how far it got, how big and how
long, and which model on which device.

### What runs it

llama.cpp, in its own process, exactly as the standalone application ran it.
GGUF weights, an optional multimodal projector, and a choice of full GPU
offload, partial offload, mixed system-RAM execution or CPU. Nothing is
converted into a PyTorch model and nothing is handed to Forge's memory manager
to move — a separate process is what makes it possible to hand VRAM back
reliably, because ending it releases the allocation with a certainty no
in-process cache can offer.

**Point it at a model you already have.** LLM Studio looks for its runtime and
weights under a data directory, in this order:

1. `PROMPT_MASTER_ROOT`, the standalone application's own environment variable;
2. the **LLM data directory** setting;
3. `<WebUI data directory>/model_chain_llm`.

The first two exist so an existing Prompt Master install is reused as-is —
runtime, weights, characters and chats — rather than downloaded again.

### First-time setup

Everything is in **Setup**, and it is two steps in order: a runtime, then a
model. There is nothing to run a GGUF with until the first one is done, so the
panel does that one first.

**1. llama.cpp runtime.** Three routes, and the panel tells you which apply:

- **Detect** — picks up a build already sitting in `<data directory>/runtime/`,
  including one left there by a standalone install you pointed at.
- **Use this runtime** — give it the path to a `llama-server` binary (or the
  folder holding it) from any llama.cpp release you already have, and it is
  copied into the data directory. This route works on every platform.
- **Download the pinned build** — fetches and SHA-256-verifies the build from
  `release-manifest.json`. Those are **Windows x64 archives only**, so this
  button is disabled elsewhere and the panel says why.

The runtime is copied in rather than referenced where it lies, because it is a
program this extension *starts* — unlike the weights, which it only reads. A
release folder is copied whole, since llama.cpp loads its shared libraries from
beside the server; a distribution-packaged `llama-server` that finds its
libraries on the system path is copied on its own, and the panel says so,
because that one stops working if the package is removed.

**2. Which model runs.** Any GGUF on disk, with or without a projector. That
changes two lines of state and downloads nothing. The **Find the projector
beside it** button suggests a neighbouring `mmproj` file — check it belongs to
that model, because nothing in a filename proves it does.

Every path box has a **Browse** button beside it, and it opens your operating
system's own file dialog, so none of this has to be typed. (If the WebUI is
running with `--listen` or `--share` it is being looked at from another machine
and a dialog opened on the server would be no use to you, so a folder browser
opens in the page instead and says why.) If you would rather paste, paste
anything reasonable: Explorer's *Copy
as path* quotes (`"C:\models\thing.gguf"`), a dragged `file://` URL,
`%USERPROFILE%\models`, or the **folder** your models are in — a folder holding
one model is not an ambiguous answer, and a folder holding six is answered by
naming them. Point it at the wrong shard of a split model and it takes the
first, because that is the one llama.cpp wants. Whatever it works out is
written back into the box, so what was recorded is always visible.

### Context, and what it costs

Context sizing, the buffer, the context size and the two key/value cache types
are on the WebUI's **Settings** page under **Model Chain**, with the residency
settings below. They describe the installation rather than any one generation,
so the host stores them with the rest of its configuration and they survive a
restart. **Setup** shows what they add up to.

Three budgets, kept separate because they behave differently:

- **context size** — tokens asked of llama.cpp;
- **context / VRAM buffer** — memory set aside for the key/value cache that
  context implies;
- **runtime reserve** — llama.cpp's scratch space, which exists whether the
  context is 512 tokens or 128k.

The panel estimates the second from the model's own GGUF header rather than from
a tokens-per-gigabyte rule of thumb, because a rule of thumb is wrong for every
model. A grouped-query model with 8 KV heads costs a quarter of what a 32-head
model of the same width costs, and the header is where that is written down.

It shows a buffer → estimated → recommended table, the model's own context
ceiling (which always wins over what VRAM would allow), and the two answers that
only matter on a shared card:

> how much context fits **while keeping the image model resident**, and how much
> becomes available **if it is demoted**.

The runtime reserve starts as a coarse allowance and is then replaced by
measurement: the first real load of a given model and placement records what the
card actually lost, and the panel says **calibrated** instead of **estimated**
from then on.

### Sharing the card

Two modes, chosen in Settings:

**Hybrid** (the default) keeps an image model and an LLM resident together
whenever they both fit. Alternating between them then costs nothing, because
nothing moves. The rule is stated as a sentence and implemented as one:

> Never unload merely because another workload started. Demote only because the
> incoming workload actually needs the memory.

**Exclusive** gives one family the whole card at a time. It is more predictable
and leaves more headroom for a single large workload, and the handover happens
when a workload starts rather than when the arithmetic forces it — ownership
that depended on the size of the last request would not be ownership.

When something genuinely does not fit, what gives ground is a policy:

| Policy | Behaviour |
| --- | --- |
| **Adaptive** (default) | Lower the LLM's context first, then move a checkpoint. A context nobody is using is cheaper to give up than a model somebody is about to use. |
| **Preserve image** | Never move the checkpoint. Shrink the LLM's context, then its GPU offload, then run it from system RAM. |
| **LLM priority** | Give the LLM the placement it asked for and demote image residency to pay for it. |

One thing is not negotiable under any of them: **an image generation always
outranks an idle LLM.** Ordinary txt2img has to keep working, so a policy that
let a background llama-server starve a generation you are watching would be a
bug rather than a setting.

Whatever gets reduced is reported. If your 128k context became 24k to fit
alongside a checkpoint, the status line says so.

**A server that is up stays up.** Placement is decided when llama-server is
started, not before every message. Once it is running, a message is answered by
the server that is already there — which is what keeps llama.cpp's prompt cache,
so a reply reads the message you just sent instead of reprocessing the whole
conversation. It is replaced when you change something (the model, the device,
the offload, the context, the cache types), when the image side takes the VRAM
back, or when the card has since freed up enough to put meaningfully more of the
model on it than it is running now. It is never replaced with a *smaller*
placement: a running server holds its VRAM whether or not it is using all of it,
so moving it into a corner of the card frees nothing — when the image side needs
that memory it says so, and the server stops.

When the image side does need the VRAM back, you choose what happens to the LLM:
stop the server, which releases every byte and leaves the weights warm in the
system page cache so a restart reads from RAM rather than disk; or keep it
running with its weights in system RAM, which avoids the reload and is much
slower to generate with.

### Taking turns

Models may share VRAM; jobs do not share the GPU. An LLM turn will not start
while the WebUI has a job running, and a generation waits — once, briefly — for
an LLM turn already in flight. The wait is bounded in both directions, so a
wedged runtime delays a generation rather than preventing one.

The one thing this deliberately does *not* do is hold a lock across a whole
generation. `postprocess` is not called from a `finally`, so a generation that
raised would leave the lock held and LLM Studio dead until the WebUI restarted —
a far worse failure than the brief overlap the lock would have prevented.

### If it goes wrong

Nothing in the LLM half can take the image half with it. The tab is built inside
a guard and renders an explanation rather than raising; a runtime that will not
start is a sentence in the status line; a reclaim that fails costs an eviction,
not a generation. If you never open LLM Studio, none of it runs — and the
Settings toggle removes the tab entirely.

## Layout

```
mc_arch.py            architecture detection + per-architecture geometry
mc_memory.py          image model residency / cache management
mc_lora.py            prepared LoRA state + stage isolation
mc_infotext.py        infotext write + paste-field registration
mc_presets.py         named Stage 2 configurations
mc_progress.py        whole-job progress model + measured timings
mc_references.py      Stage 2 supplemental reference routing
mc_styles.py          style library integration helpers

mc_broker.py          cross-workload residency policy and the workload lock
mc_gguf.py            GGUF metadata header reader
mc_llm_context.py     context capacity estimation and its calibration
mc_llm_runtime.py     the managed llama.cpp process and its placement
mc_llm_paths.py       where LLM Studio keeps its data
mc_llm_files.py       what a pasted path means, and what is in a folder
mc_llm_browse.py      the Browse button beside every path box
mc_llm_native.py      the operating system's own file dialog
mc_llm_setup.py       getting a llama.cpp runtime in place
mc_llm_state.py       shared preferences + the two mode histories
mc_llm_sessions.py    the three run orchestrations, as streaming generators
mc_llm_studio.py      the LLM Studio tab shell, model chooser and Setup mode
mc_llm_prompt_panel.py     Prompt Studio workspace
mc_llm_chat_panel.py       Conversation workspace
mc_llm_minimax_panel.py    MiniMax H3 workspace
mc_llm_ui.py          shared UI helpers and the element-id contract
prompt_master/        vendored LTX business logic (see VENDORED_FROM.txt)

scripts/model_chain.py  Script class, UI, orchestration
style.css             optional progress-bar appearance + LLM Studio styling
javascript/           the settings-to-CSS layer, and LLM Studio polish
tests/                pytest suite (runs without a WebUI)
docs/                 revised specifications for the progress and LLM work
```

The LLM modules stack in one direction and never the other:

```
mc_llm_*_panel  ->  mc_llm_sessions  ->  mc_llm_runtime  ->  mc_broker
       |                                      |                  |
mc_llm_browse  ->  mc_llm_native        mc_llm_context      mc_memory
       |                                      |
mc_llm_files  <-  mc_llm_setup            mc_gguf
```

`mc_memory.py` does not import `mc_broker`, and that is deliberate rather than
incidental: the image half stays importable, testable and correct on an
installation that never loads the LLM half. What it has instead is one optional
hook, installed by `mc_broker` at import, which it calls when Forge's own
eviction has fallen short. It never decides *whether* another workload should
give ground — that is the broker's to hold.

`mc_lora.py` deliberately depends on nothing else in the extension. It is a
description of two host mechanisms — the extra-network tag syntax and the LoRA
hash on `sd_model` — and keeping it free of settings lookups and cache
internals is what makes it testable against the host's behaviour rather than
against ours.

Two deviations from the layout in the design document, both forced by how Forge
Neo loads extensions:

- **Helper modules live at the extension root, not in `scripts/`.** `load_scripts()`
  imports *every* `.py` file in an extension's `scripts/` directory as a separate
  script module, so helpers placed there would be loaded twice under two module
  identities — fatal for a module holding cache state. The extension root is what
  gets added to `sys.path`, so root-level helpers import cleanly as top-level
  modules. This is the same structure the bundled `sd_forge_lora` extension uses.
- **`mc_arch.py`, `mc_presets.py` and `mc_references.py` are additional
  modules**, for architecture detection (needed by both the UI and the
  orchestration code), preset storage, and Stage 2 reference routing.

`mc_references.py` is the only module that knows ImageStitch exists, and it
reaches for exactly one thing: the contents of the user's input gallery, read
out of `p.script_args`. It never calls ImageStitch's `process()`, never touches
its cached parameters as a data source, and never writes to its gallery.

`mc_memory.py`'s two-slot cache is private to that module. `ensure_resident()` /
`get_model()` expose no slot count, so swapping in an LRU pool for 3+ models
needs no change to the orchestration code.

## Tests

```
pip install pytest pillow numpy psutil httpx
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
from a damaged store, encoder pinning and its fallbacks, the background Stage 1
preload and its failure handling, prepared-LoRA-state reuse and every case that
invalidates it, stage isolation of extra-network tags, the VRAM reserve and its
four floors, speculative warming of Stage 2, interruption handling, Stage 2
reference routing in all three modes with its ordering, capability and cleanup
rules, whole-job progress and its measured calibration, the UI's control-order
contract, and inertness when disabled.

The LLM half adds: the GGUF metadata reader against synthetic headers and every
way one can be malformed; per-model context arithmetic, including the
grouped-query case a constant would get wrong; the model's own context ceiling;
the calibration that replaces the estimated runtime reserve and survives a
context change; the residency policy in both modes and all three placement
policies; rank protection for active and pinned residency; workload
serialisation and the bounded waits either side of it; placement negotiation and
the requirement that every reduction is reported; the estimator preview being
free of side effects; the two mode histories staying separate files; the three
run orchestrations and their event sequences; the panels assembling with their
control lists in agreement; and the theme contract — extension-owned element
ids, no hard-coded colours, no Gradio-generated selectors.

Three of those files are lopsided on purpose, because their failure modes are:

```
tests/test_lora_state.py       reusing prepared state that is no longer valid
                               produces a wrong image and no error anywhere,
                               so the invalidation cases outnumber the
                               preservation ones
tests/test_warming.py          warming into the reserve costs an OOM or, worse,
                               a silent spill to system RAM, so most of the file
                               is about warming *not* happening
tests/test_residency_speed.py  a preload that fails must cost one generation
                               rather than every one
```

One invariant in `mc_memory.py` is worth stating outright, because breaking it
cost a user every generation until they restarted the WebUI: **the cache may
release its own reference to a model, never a caller's.** `drop()` pops the
entry and stops there. It must not blank `entry.sd_model`, because an entry
handed out by `get()` is a live object someone may still be holding — and
someone is. `reinstate_pending()` looks its entry up, then stashes the outgoing
Stage 2 model, and that stash can evict the very entry it is holding: the two
models are the two largest things in the cache, and Stage 2's arrival is exactly
when the budget runs out. Blanking made the swap back install nothing, leaving
the host with no model and `forge_hash` still asserting there was one — which
`forge_model_reload()` believes, so every later generation died the same way.

The criteria that need real hardware — that an SDXL → Flux.2-Klein chain
produces coherent output, that a Krea 2 Edit refine responds to its Edit LoRA,
that a LoRA visibly affects the refined image, that a warm switch is measurably
faster than a cold disk load, that a preserved LoRA state survives 20–50
alternating jobs without drift, and that multiple Stage 2 references visibly
condition a Flux.2 Klein or Krea 2 + Edit LoRA refine — are left to manual
verification.
