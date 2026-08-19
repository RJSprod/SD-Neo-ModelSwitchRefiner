

<img width="335" height="584" alt="Screenshot 2026-07-25 192828" src="https://github.com/user-attachments/assets/773b2f5c-720d-4715-9fe5-5257de2b09c3" />



https://github.com/user-attachments/assets/e05ff7e0-f384-41b0-ad32-1259ec00f05b




# 🎬 Prompt Master LD

A ComfyUI node that writes **LTX-Video 2.3** prompts with a local LLM — accent, wardrobe, camera, music, visual style and first-person POV, all driven from one panel and all resolved into a single flowing shot script.

Not a template filler. The node hands a 26B-class local model a **brief about how LTX actually behaves**, then gets out of the way. Every dropdown you leave alone costs zero tokens.

> ### A note before you start
>
> This is a hobby. I build these tools for my own personal fun and I'd be making this
> one whether anyone else ever used it or not. — I share it because that part is fun too.
>
> It has been hammered hard on my own machine but a node this size will have bugs I have not hit yet: different GPUs,
> backends, workflows and models. If something breaks, open an issue and I will
> have a look — tell me your backend, your model, and what you had selected.
>
> Feature requests, accent suggestions and prompt findings are all welcome. A lot
> of what is in here came from spotting something wrong in a render and chasing
> it down, so if you spot one, I want to know.
>
> — LoRa-Daddy

```
🎬 Prompt Master - LD  →  pack  →  📦 Prompt Unpack - LD  →  image · reference · positive · negative · width · height · fps · frames
🧱 Lora Loader - LD    →  model · clip
```

---

## Why it's different

LTX 2.3 is **literal**. Anything you name appears; anything you omit renders as a generic average. The whole brief is built on that one fact:

| principle | what it means in practice |
|---|---|
| **Physics, not vibes** | "she looks sad" renders nothing. "her jaw sets, eyes dropping to the floor" renders. Every law is written in what a camera can see. |
| **No examples, ever** | Give a model example dialect words and it parrots them into every shot. The brief describes *what to do* — never a word to copy. Enforced by the test suite. |
| **Conditional laws** | 47 accents, 35 genres, 20 styles — but only the *one* you picked is ever sent. A bare text-to-video brief is ~625 words; a fully loaded first-person shot is ~2,800. |
| **Budget by the clock** | Word count, beat count and spoken-line count all scale from your actual duration, and dialogue is capped by how long a line takes to *say*. |
| **The request wins** | Your intent outranks every dropdown. An attached still outranks every dropdown. Dials only fill in what you left open. |

---

## What's in the box

### 🗣 47 accents — voice, not caricature

Each accent is one line of phonetic direction: which sounds bend, how, and where the stress lands. The model respells the dialogue accordingly.

| region | accents |
|---|---|
| **North America** | American · New York · Californian · Midwest US · Boston · Canadian · Southern US · Appalachian · AAVE |
| **British Isles** | RP British · Cockney · Scottish · Irish · Scouse · Geordie · Northern English · Welsh |
| **Caribbean** | Jamaican · Trinidadian |
| **Africa & Middle East** | Nigerian · Ghanaian · South African · Swahili · Arabic · Hebrew |
| **Asia** | Indian · Filipino · Korean · Japanese · Mandarin · Thai · Vietnamese |
| **Europe** | French · Spanish (Castilian) · Spanish (Latin America) · Italian · Portuguese (BR) · German · Dutch · Swedish · Norwegian · Russian · Polish · Czech · Greek |
| **Oceania** | Australian · New Zealand |

**Two kinds, two sets of rules.** This distinction matters more than the list itself:

- **Second-language accents** (French, German, Russian, Mandarin…) — the target is *English spoken with that accent*. Native vocabulary is contamination, so it's capped at one short slip.
- **Varieties of English** (Jamaican, Cockney, Nigerian, AAVE, Appalachian…) — here the lexicon, idiom and grammar **are** the accent. Capping them removes the only recognisable part, so these get the opposite instruction: use this variety's own words and sentence shapes freely.

**Accent strength: natural · strong · thick.** Turning it up scales four things — how much of each line is respelled, whether sentence shape follows the accent or standard word order, how often its address terms and tags appear, and whether the accent reaches the *non-words* (the hesitation sound, the laugh, the breath). One hard limit at every level: a respelling has to stay a word a person could read aloud, because spelling mangled past recognition renders as text burned across the picture instead of sound.

**No accent selected still describes a voice** — register, grain and pace — because an unnamed voice comes back as flat synthetic narration.

### 🎵 35 music genres, and one that picks itself

Described by **sound**, never by song title: instrumentation, tempo, where the beat lands. Bodies then move to it.

**Set "Music — auto" and it follows your accent.** Jamaican → dancehall. Trinidadian → soca. Nigerian → afrobeats. Appalachian → bluegrass. Every accent maps to something real.

`House / club` `Hip-hop` `Trap` `Techno` `Drum & bass` `R&B` `Pop` `Rock` `Reggaeton` `Disco` `Jazz-funk` `Ambient` `Lo-fi` `Classical / orchestral` `Cinematic score` `Opera — carried voice` `Opera — pure harmonics` `Reggae` `Dancehall` `Soca` `Afrobeats` `Amapiano` `Bollywood / filmi` `Bhangra` `K-pop` `City pop` `Flamenco` `Tango` `Samba` `Celtic trad` `Country` `Bluegrass` `Grime` `Shaabi` `Chanson`

The two opera modes exist because they're different instruments — and each carries its own lip-sync rule. A sustained vowel is the easiest thing a mouth can hold, so the aria says *sit on her face while she holds the note*. Runs faster than breath can't be synced at all, so pure harmonics says *keep the frame off the mouth and save it for the long notes*. **`background`** ducks any genre under the scene as ambience.

### 🎨 20 visual styles

The style is declared in the script's opening words — arrive late and LTX reverts to photoreal. Each entry carries its own grammar, its own forbidden list, and a per-beat anti-drift clause, because a 16-second clip creeps back toward photography.

| group | styles |
|---|---|
| **Cartoon universes** | SpongeBob · Rick and Morty · Lego Batman |
| **Animation** | Pixar / 3D · Anime (cel-shaded) · Bold-outline cartoon · Flat vector · Claymation · Stop-motion · Watercolour storybook · Comic book |
| **Live action** | Film noir · Neo-noir / neon · 70s analog film · VHS camcorder · Documentary realism · Music video · Horror · Western · Fashion editorial |

Crucially the style **absorbs** the rest of your settings rather than replacing them: pick an accent, wardrobe, music and camera under a cel-shaded style and you get all of them, rendered as animation. Image-to-video ignores the style dropdown entirely — the attached still already *is* the medium.

### 🎥 10 camera behaviours · 10 transitions

`handheld restless` `hunting` `arm's-reach POV` `slow push` `slow pull` `orbit` `locked off` `slow rise` `circling close` `float`

Each overrides the generic single-move rule and demands a named camera beat in every paragraph. Transitions: `morph / melt` `hard cut` `whip pan` `match cut` `push through detail` `smash zoom` `dissolve` `spin blur` `flash cut` `pull-back reveal` — each with matched negatives that won't fight the positive.

### 👁 First-person POV, properly

The hardest thing to get right in AI video, and the most developed system here. The camera is the viewer's eyes — no face, no torso, no name, no body walking into shot.

- **Hands are the only visible part of the viewer**, and they get their own law: how many (and held to that number), what they're doing (a hand touching nothing renders as a floating object), which way they face (back of the hand toward the view — an open palm turned at the camera is the pose that comes back melted), and whose they are.
- **The viewer's sex rides on the hands** — "a woman's hands" inside the opening two sentences, never as a person standing there, because a bare gendered noun that early summons a *body* into frame.
- **Who speaks is decided by physics.** Only a mouth on screen can lip-sync, so the person in frame carries most of the dialogue and the viewer's off-screen lines stay the minority — unless nobody else is in the shot, in which case the viewer is the only voice and *inventing* a companion to talk to is banned outright.
- Contact, orientation, mirrors and hand-count are all handled; a mirror reflection is the one place the viewer's body may legitimately appear.

### 👗 Wardrobe · 🧬 identity · 🔞 undress

Seeded garment, palette, fabric and one lived-in detail — biased to `her`, `him` or auto-detected from your intent, with the piece always outranking the palette. Identity seeds a plausible look from the accent's region so the cast isn't generic, and **both read the same detector**, so a shot about a man never gets a woman written into sentence one. Clothes are treated as objects with weight and drape, not adjectives. Optional undress chain with a real four-stage sequence, plain anatomy words, and no garment teleporting back on.

### ✍ Two output formats

- **Flowing prose** (default) — blank-line beats, no labels, no timestamps. What LTX's own guidance asks for.
- **Shot script** — bracketed time windows for tighter control of *when* things happen.

Both budget identically, so a same-seed A/B between them is a fair comparison.

### 🚫 Smarter negatives

Content-aware banks that arm themselves from your settings, with one rule: **every term names something a frame could actually contain.** "Slideshow" and "teleporting between poses" describe a narrative failure, not pixels — a sampler has no embedding for them. They're replaced by the artifact they actually look like (`frozen frame`, `motion smear`, `duplicated limbs`, `camera in frame`). Everything is de-duplicated on the way out, because a term stated twice weights that concept twice.

**Optional smart negative** runs a second, cheap LLM pass over the *finished* script and names the visual opposite of what the shot committed to — night shot gets `daylight`, cel animation gets `photorealistic skin`, handheld gets `locked tripod frame`. Guarded three ways: any term whose words appear in the script is dropped, anything over four words is dropped, and a failed pass costs you nothing.

### 🧱 Lora Loader LD

Ten slots, live enable/disable, model and clip in one node. Scroll the strength fields to nudge them.

---

## Install

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/Brojakhoeman/Prompt-Master-LD.git
```

Restart ComfyUI. The nodes appear under **LD/PromptMaster**.

No Python dependencies beyond what ComfyUI already ships.

`cpld_conn.json` holds your local backend settings (model paths, server URL). It
is written on first run and should stay out of version control — add it to
`.gitignore` if you fork.

### Pick an LLM backend

Click the **⚙** cog on the node. Three options:

**llama.cpp (managed)** — the node launches and kills the server itself.
- Put `llama-server.exe` in `C:\llama\` (Windows) or on your `PATH`
- Point **models_dir** at your GGUF folder
- Pick a model, and optionally an **mmproj** file for vision (needed for image-to-video grounding)
- Recommended: https://huggingface.co/HauhauCS/Gemma4-26B-A4B-Uncensored-HauhauCS-Balanced/tree/main

**LM Studio** — `http://localhost:1234`
- Load your model in LM Studio's own UI first, then hit **Health** in the cog panel

**Ollama** — `http://localhost:11434`
- Ollama running locally; pick the model by name

**Set `ctx` to 16384 if you use long clips with every dropdown on.** A fully loaded brief plus a 30-second completion budget will squeeze an 8192 context. The node clamps and says so in the status line rather than truncating silently.

---

## Quick start

1. Drop a **🎬 Prompt Master - LD** node in and wire `pack` → **📦 Prompt Unpack - LD**
2. Wire the unpack outputs into your LTX 2.3 sampler (`positive`, `negative`, `width`, `height`, `frames`)
3. Pick **I2V** or **T2V**. For I2V, drop a start frame on the thumbnail
4. Set aspect and size — the frame count snaps to LTX's `8n+1` automatically
5. Choose what you care about, ignore the rest
6. Type your intent: *what happens in the shot — subject, place, the one thing that changes*
7. Hit **▶ Generate**. The script streams into the box below
8. **Edit it freely** — that exact text is what queues

### The Talk dial has three zones

| setting | what you get |
|---|---|
| **0** | Silent. No quoted voice, no mouths speaking. |
| **1–69** | Dialogue. Lines capped by the clock — a quoted line takes 2–3 seconds to say, so a 12-second shot holds about four at most. |
| **70–100** | **Performed vocal.** The voice becomes the through-line, the stop-before-speech rule relaxes, and the body performs around it. This is where singing and rap live. |

Past 50% the panel warns you, and the line count stops climbing rather than asking for twenty seconds of speech in a twelve second shot.

### Seed

The panel's **▶ Generate** bypasses ComfyUI's queue, so it advances the seed itself — at the top of the click, so the number on screen is the one that wrote the script. Modes: `randomize` (default), `increment`, `decrement`, `fixed`. **Use `increment` for A/B testing** — the walk is reproducible, so a good seed can be found again. 🎲 rolls once without changing the mode.

### Fast re-roll

Tick it and the LLM stays resident between generates instead of reloading each time. VRAM is released the moment you queue the render, so LTX still gets the whole card.

---

## Panel reference

| control | notes |
|---|---|
| **I2V / T2V** | I2V grounds on the attached still; T2V builds the world from nothing |
| **aspect · W/H · size** | LTX 2.3 presets, live megapixel readout, `8n+1` frame snapping |
| **POV** | off · ♂ · ♀ — sets the viewer, and the viewer's hands |
| **Talk** | 0–100, three zones (above) |
| **Accent + strength** | 47 accents · natural / strong / thick |
| **Camera · Transition** | 10 each |
| **Music + background** | 35 genres plus auto; background ducks it under the scene |
| **Wardrobe** | auto / off / her / him, plus undress |
| **Lexicon** | Add a character name or LoRA trigger word and it's carried into the script verbatim |
| **Style** | 20 visual styles (T2V only) |
| **Format** | Flowing prose or shot script |
| **fps · sec** | Or wire them in from upstream |
| **negatives** | Extra terms, plus the smart-negative toggle |
| **⚙ cog** | Backend, model, mmproj, ctx, theme, VRAM free |

Box heights persist across reloads. Themes live in the cog.

---

## Under the hood

```
PromptMasterLD/
├── __init__.py      # node registration + WEB_DIRECTORY
├── node.py          # the two nodes, widget order, pack assembly
├── brain.py         # assembles the brief — the core reasoning
├── routes.py        # /cpld/* endpoints, streaming, VRAM hand-off
├── backend.py       # managed llama.cpp / LM Studio / Ollama
├── accents.py       # 47 accents, variety-vs-L2 classification, strength
├── music.py         # 35 genres, accent pairing, lip-sync notes
├── styles.py        # 20 styles, absorb + anti-drift laws
├── cinematics.py    # camera behaviours and transitions
├── hands.py         # POV hand law
├── identity.py      # seeded cast look by region
├── wardrobe.py      # garment / palette / fabric seeds, undress chain
├── negative.py      # negative banks + the smart pass
├── shotscript.py    # output contract, both formats
├── imaging.py       # still handling, thumbnails, vision encode
├── vram.py          # flush + cache purge
├── lora_loader_ld.py
├── selftest.py      # 429 assertions
└── js/              # panel UI, lora UI, themes
```

### API routes

| route | purpose |
|---|---|
| `POST /cpld/generate` | streaming generate (SSE) |
| `GET /cpld/health` | backend status |
| `POST /cpld/backend` | switch managed / remote |
| `POST /cpld/models` | scan GGUFs |
| `POST /cpld/free` | release VRAM now |
| `POST /cpld/upload` · `GET /cpld/thumb` · `GET /cpld/imginfo` | image handling |
| `GET /cpld/lora_list` · `POST /cpld/lora_keycounts` | LoRA discovery |

### Test suite

```bash
python selftest.py
```

**429 assertions.** Not unit tests — *interaction* tests. They check that the right laws are present, the wrong ones absent, and that no two laws contradict each other in any reachable configuration: POV never gets a face, the style law never claims the viewer's invisible body, a silent shot has no voice rules, a male subject never gets a female wardrobe detail, no accent smuggles in an example word, and the dialogue count always fits the clock. Plus a sweep across every reachable combination of mode × POV × format × talk × duration × style × strength × accent.

Run it after any edit to the brief. It has caught more real regressions than reading ever did.

---

## Troubleshooting

**UI not loading** — hard refresh (Ctrl+Shift+R), then restart ComfyUI fully. Check the browser console.

**LLM won't connect** — hit **Health** in the cog for the actual error. Confirm the server is up and a model is loaded on the remote side.

**Script looks truncated** — check the status line for a token-ceiling message and raise `ctx` in the cog.

**Old workflow loads with the seed stuck on "fixed"** — earlier builds wrote that value in. Change the dropdown once per workflow.

**Text burned across the video** — the accent respelling is too aggressive. Drop accent strength from `thick` to `strong`.

**Flat, robotic voice** — no accent selected, or the voice was never described. Pick an accent, or describe the voice in your intent.

---

## Credits

**Prompt Master LD** by LoRa-Daddy — *paulhaul / Brojakhoeman*

Built on ComfyUI, llama.cpp, LM Studio and Ollama, for Lightricks' LTX-Video 2.3.

Successor to [Prompt Forge LD](https://github.com/Brojakhoeman/Prompt-Forge-LD).

## License

MIT — use it, fork it, ship it.

---

*Status: active development · July 2026*
