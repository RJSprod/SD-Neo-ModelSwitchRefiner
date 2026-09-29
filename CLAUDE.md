# SD-Neo-ModelSwitchRefiner

A Forge / SD WebUI Neo extension. Four areas share one process and, critically, one GPU:

- **Model chain and refiner** — multi-stage image generation, checkpoint switching, Krea 2 and Flux Klein support (`mc_arch.py`, `mc_plan.py`, `scripts/model_chain*.py`)
- **LLM Studio** — local `llama-server` chat with vision, attachments, roles (`mc_llm_*.py`)
- **Voice Chat** — STT and TTS with several selectable engines (`mc_voice_*.py`, `voice/`, `*_worker/`)
- **Creative / spatial prompt tooling** — treatments, bounding-box regions, literal prompts (`mc_creative_*.py`, `mc_spatial*.py`, `prompt_master/`)

`README.md` is long and acts as the behavioural specification. Treat it as authoritative for declared behaviour and update it when behaviour changes.

The hard problem running through the whole codebase is **VRAM arbitration**: the image model and the language model want the same card, and most of the subtle bugs in this project's history come from that contention. Since handoff 28 there is a third party to it: Mini Paint NEO's WanGP, on a card of its own, which `mc_wangp.py` gives priority over the language model on that card. Since handoff 31 there is a fourth, not yet installed: a GPU speech guest (VibeVoice 7B, for the Voice Box tab, `docs/23-voice-box.md`) that takes a card only through a turn — `mc_turns.py` decides whose turn it is on each card, holds image jobs at a gate while one is next, parks and brings back the image model, and borrows WanGP's card from Mini Paint through a lease.

---

## Project history lives in Claude artifacts

Sessions from 2026-08-20 onward each published a detailed handoff as a **private Claude artifact** titled `Handoff NN - <topic>`. They contain turn-by-turn logs, design decisions, rejected alternatives, and gotchas.

**They are not loaded automatically.** To use them:

1. `Artifact` tool, `action: "list"` — the handoffs appear as `Handoff 01` … `Handoff 30`
2. `action: "read"` with the URL of the one you need

**Read the relevant handoff before reopening an area it covers.** The index below tells you which one.

### Coverage limits — read this before assuming

- **Numbering gaps at 06 and 19 are not missing records.** Those sessions targeted other repositories (`SD-ForgeNeo-ExtendedLoraDetails` and `NEO-webui-auto-tls-https`); their handoffs live there.
- **The handoffs do not cover the whole repository.** PRs #1–#44 predate the handoff record entirely; PRs #82–#102 (prompt-box UX, QoL/UX refactor, pipeline header, design-intent spec review) fall between handoffs 14 and 15 with no handoff written; and PRs #192–#223 (the Forge Assistant flyout, rounds one to four: read-aloud, message actions, auto-attach, literal prompt boxes, new thread, system-prompt editor, in-place edit, Send to Generate) fall between handoffs 27 and 28 with no handoff written — handoff 28 names them only in its state section. Absence from this index does **not** mean the work never happened — check `git log` before concluding anything is unimplemented.
- **Handoff 01's internal date is wrong.** It states 2026-09-03; the work was 2026-08-20. Later handoffs carry correct dates.
- **Sessions under-report their own PRs.** Several handoffs say "no PR opened" because the *user* opened it from the UI rather than the session. The PR numbers in the index below come from `git log` and are authoritative.

---

## Superseded work — check here before rebuilding anything

The record is chronological, so an older handoff can describe an approach that was later torn out. These are verified against the code at HEAD, not merely taken from what each session claimed.

| What | Where it came from | Current reality |
|---|---|---|
| `_arm_llm()`, `_swept_by_generating()` in `mc_arm.py` | Built in **16** | **Removed in 25.** Zero references at HEAD. Handoff 16's arming architecture is stale; handoff 25 is authoritative. Commit `8f255f5` was itself superseded by `72fbad6`, which deleted the function. |
| Pinned LoRAs control | Present through **10** | **Removed in 11**, deliberately and in one step rather than deprecated in phases, at the user's request. New prompt syntax supersedes it. Zero references at HEAD. |
| Creative mode radio + pinned dropdown + exclusion list | Built in **04**, **07** | **Replaced in 14** by a single treatment multi-select (`apply_treatments`). The old controls are superseded *on the surface*; the machine-facing values underneath were kept. Handoff 04 and 07 UI descriptions no longer match the screen. |
| Klein spatial **region geometry** | Built in **13**, PRs #74–#78 merged | **Reverted at the user's request.** It cost roughly five model evaluations per region per step, which the user judged unacceptable. Do not rebuild without solving that cost first. Note the distinction: Flux Klein as a *checkpoint/architecture* is alive and well in `mc_arch.py`. Only the region geometry was undone. |
| Klein prompt dialect (`DIALECT_FLUX2`) | Built in **12** | **Never merged.** Branch `claude/affectionate-gates-b1ob6o` has no merged PRs and the symbol is absent at HEAD. The work exists only on that branch. |
| Reclaim threshold knob | Proposed in **27** | Superseded by the floor rule before it was built. |
| Workspace scroll correction | Added in **15** | Reverted later in the same session (turn 4). |
| PR #63 | **09** | Closed as superseded by #64–#69. |
| Editing a message in the flyout's own composer box (`editBar`, `applyEditing`, Send reading Save) | Round four, PRs #192–#223 | **Replaced in 29's UI round** by the message editor dialog, `javascript/mc_message_editor.js`, which LLM Studio's Edit opens too. Zero references to `editBar` or `applyEditing` at HEAD; the flyout's `editing` holds only the target. |
| Focus button in the flyout header; Conversation accordion heading; the character · thread selector line | Header row since round one | **Removed in 29's UI round.** Three presses on the header's drag space toggle focus (`tapHeader`); the **Chat** glyph does what the heading did (`applyChat`); the character's name heads the ⋯ menu's Chat group and the thread list is its Threads submenu. |
| The flyout's menu fixed to the window (`placeMenu`, `MENU_GAP/EDGE/FLOOR`) | Round two, design doc §3.14 | **Replaced after 29's UI round** by a menu inside the panel in the conversation's place (`applyMenu`). Zero references to `placeMenu` at HEAD. |
| "The tab stays on the thread it is on" when the panel chooses or starts a thread | Round three (New thread from the ⋯ menu) | **Reversed after 29's UI round**: the panel and the tab are one conversation both ways. The panel presses the tab's open bridge (`openInStudio`); a look follows only a *move* of the tab (`Store.noteTab`). |
| "WanGP's VRAM is priority one and is never taken from it" as an absolute | **28** | **Amended in 31**, not reversed: a VibeVoice turn takes WanGP's card *between* WanGP's jobs, by WanGP's leave, through Mini Paint's lease (`mc_wangp.lease_*`, `minipaint_neo.wangp.turns`, bridge 1.12.0). The language model's side of handoff 28 is unchanged: it still never takes WanGP's VRAM. |
| The sliding-window cache kept on the Intel GPU by default (`_on_intel`) | **29** | **Reversed in 30**: *Automatic* is the full cache on every device. The window's checkpoints split each turn into 3.6 batches on average, at about 1.5 s each on SYCL with a mixture-of-experts model, and a saved prompt cache resumes exactly only on the full cache. The window stays a choice, with `--checkpoint-min-step 2048`. No definition or call of `_on_intel` at HEAD (test names that end in `_on_intel` are unrelated). |

### Shared with Mini Paint NEO — do not rename

The class `forge-assistant-focus-root` (put on the focused panel by `forge_assistant_focus.js`) is read by Mini Paint NEO's `style.css` and `browser/minipaint_wangp.js` to make its WanGP tab the frame alone under focus mode. Renaming it here breaks that silently. The other shared things are older: the dialog layer `--minipaint-dialog-layer: 2000`, the `minipaint:overlay` event, and `visible()`'s reading of a `visibility: hidden` panel as not showing (a parked WanGP panel). And one newer, since handoff 31: **the card lease**, `minipaint_neo.wangp.turns` (`request`/`state`/`flush`/`release`/`report`, a record of exactly `LEASE_KEYS`), which `mc_wangp.lease_*` calls and `mc_turns` drives. Its key set is held closed on both sides (`mc_wangp.LEASE_KEYS` reads a record missing a key as no answer), its `flush` is asynchronous (measure after `held`, never after `flush` returns), and a lease that is `holding` with `bridge_hold: unsupported` is refused rather than waited on. Mini Paint's `docs/wangp/CONTRACTS.md` is its other half.

### Not superseded — a trap in the other direction

**The three TTS engines coexist.** Kokoro (17, 18), Sopro V2 (22) and PocketTTS (23, 24) are all live, selectable backends registered in `mc_voice_engines.py` via the `ENGINES` tuple. Later engines did **not** replace earlier ones. Do not "clean up" Sopro or Kokoro as dead code. LavaSR (24, 26) is the Voice Pipeline's bandwidth-restoration stage (it rebuilds the top of the spectrum a small TTS model cannot generate, after DPDFNet), not a fourth TTS engine. VibeVoice (31) will not be a fourth Voice Chat engine either: it is a GPU guest for the Voice Box tab, and every Voice Chat engine stays CPU-only.

---

## Chronological index

Dates are session start dates. PR ranges are from `git log`.

| # | Date | Topic | Branch | PRs | State |
|---|---|---|---|---|---|
| 01 | 08-20 | Repo review; llama-server OOM race, retry ladder | `repo-review-extend-6c30tm` | — | pushed |
| 02 | 08-20 | Krea creative mode focus / browser gate removal | `adoring-mayer-hmny3b` | #45 | merged |
| 03 | 08-20 | Model selection + smart downloading, staging/resume | `model-selection-download-f7jdnz` | #46 | merged |
| 04 | 08-21 | Krea 2 creative QoL; SPREAD profile; speed metrics | `krea2-creative-qol-lctilo` | #47–#53 | merged |
| 05 | 08-21 | LLM Studio models; expert offload (`--n-cpu-moe`) | `llm-studio-models-speed-7z4wkf` | #54 | merged |
| 07 | 08-21 | Krea 2 BBOX mode review; CSS colour contract | `krea2-bbox-mode-review-kooyln` | #55 | merged |
| 08 | 08-21 | Long chain + memory spec; `_spendable()` wraps `_free_vram()` | `long-chain-memory-spec-fd8m00` | #56–#61 | merged |
| 09 | 08-22 | Model chain timeline delays; prompt reordering for cache | `model-chain-timeline-delays-oo2q0e`, `role-specific-llm-config-2k9x1p` | #62, #64–#69 | merged |
| 10 | 08-22 | Creative BBOX UI refactor; spatial decoupled from creative | `creative-bbox-ui-refactor-vjpqj8` | #70–#72 | merged |
| 11 | 08-23 | Image model content handling; **Pinned LoRAs removed** | `image-model-content-handling-ztlkhm` | #73 | merged |
| 12 | 08-23 | Flux Klein dialect detection, saved layouts | `affectionate-gates-b1ob6o` | — | **unmerged** |
| 13 | 08-23 | Klein spatial regions — built, merged, then **reverted** | `spatial-layout-flux-klein-9lqxnn` | #74–#79 | **reverted** |
| 14 | 08-24 | Text-2-image tab UX; **treatment multi-select** | `text-to-image-tab-refactor-nyfyqu` | #80–#81 | merged |
| 15 | 08-27 | LLM vision lazy loading; attachments, in-place edit | `llm-vision-lazy-load-xrpgm1` | #103–#106 | merged |
| 16 | 08-28 | Resource-scoped concurrent image/LLM; `mc_arm.py` created | `resource-scoped-concurrent-execution-4erkkt` | #107–#111 | merged |
| 17 | 08-28 | Local CPU-only voice chat; engine installer | `local-cpu-voice-chat-px9tl6` | #112–#118 | merged |
| 18 | 08-28 | Voice Chat V1.1; voice bank, cloning, worker protocol | `voice-chat-v1-1-features-hpg51d` | #119–#122 | merged |
| 20 | 08-29 | Voice STT/TTS updates; speech vs annotation detection | `voice-chat-stt-tts-updates-ok8sm8` | #123 | merged |
| 21 | 08-29 | Voice latency; NumPy pinning, gesture tokens | `voice-chat-latency-wiv92p` | #124–#126 | merged |
| 22 | 08-30 | Sopro V2 TTS backend; voice cloning, ETag/digest fix | `sopro-v2-tts-backend-ul1e60` | #127–#141 | merged |
| 23 | 08-31 | PocketTTS engine; streaming units, cancellation | `pockettts-voice-engine-up2ss4` | #142–#153 | merged |
| 24 | 08-31 | PocketTTS low-latency pipeline; LavaSR adapter | `pockettts-low-latency-voice-u32dbk` | #154–#170 | merged |
| 25 | 09-01 | Image model VRAM perf; **`_arm_llm` removed** | `image-model-vram-perf-wrog01` | #171–#172 | merged |
| 26 | 09-02 | Lava install compatibility; wheel closure hashing | `lava-install-compatibility-p9jp6m` | #173–#187 | merged |
| 27 | 09-02 | Python crash investigation; Forge unload flag / warm-up | `python-crash-investigation-n5zovf` | #188–#191 | merged |
| 28 | 09-26 | **WanGP-aware LLM placement**; `mc_wangp.py` (ceiling, watch, thread cap); Mini Paint NEO presence contract | `modest-thompson-cutso3` | #224 | merged |
| 29 | 09-26 | Vision projector **loaded from the start** (setting); prompt front moves in steps, **one still** (`TRIM_STEP`, `stills_carried`); flyout **asks after a reply until its first word**, deadlines on every request, idle looks; sliding-window cache **kept on the Intel GPU** (setting, `--checkpoint-min-step 2048`); every-picture toggle and the 768-pixel cap; **UI round**: glyph header with a **Chat** button (accordion and character · thread line gone), **Focus by three presses** on the header (button gone), 36 px rows, the **message editor dialog** (`mc_message_editor.js`, shared with LLM Studio), the **grouped ⋯ menu** with New chat and a Threads submenu, literal boxes with in-box placeholders; then **New chat moves the tab too** (open bridge `open-at`/`open-now`, `Store.noteTab`: a look follows a *move* of the tab only), **menus inside the panel** (`applyMenu`), picture cap **640** | `modest-thompson-cutso3` (reset onto the merged default after each merge) | #225–#230 | #225–#230 merged; then a **two-repository feature** (Mini Paint NEO's WanGP tab is the frame alone under focus mode, keyed on `forge-assistant-focus-root` — the class is a contract now, §3.24 of docs/22; and the WanGP **theme hijack**, bridge 1.9.0, all on the Mini Paint side): Mini Paint NEO #106 and, docs only, #231 here; then **leaving the transcript's end is a gesture** (`leaveBottom`, `scrolledTranscript` re-follows only downward; §3.25) and the **character screen relaid out with no prose** (§3.25): #232 here, and the theme's second round as Mini Paint NEO #107 |
| 30 | 09-28 | **First-word latency, from the logs**: the wait is llama-server reading the prompt on the Intel GPU (25 to 90 tokens a second); Gemma 4's template strips the empty thought marker from every past reply, so each turn re-reads the last reply; on SYCL a mixture-of-experts batch costs about 1.5 s before its first token, and the window cache splits a turn into 3.6 batches on average. Six levers were put to the user; **three dismissed by the user**: preferring the 5090 while WanGP is idle, starting the Intel server at Forge start, and faster flyout polling. Built: **saved prompt caches** (`mc_llm_slot_cache.py`): `--slot-save-path`, the answering slot saved after every reply (`verbose` + `response_fields: ["id_slot"]`), read back before a conversation's first reply on a fresh server; a model with a sliding window is saved **only on the full cache** and read back **only when the new prompt extends the saved tokens exactly** (the slot file's packed tokens against `/apply-template` + `/tokenize`), because a b10621 slot file keeps just the window. **Gemma's thought marker kept** (`mc_llm_template.py`): a patched copy of the model's own template as `--chat-template-file`, so a turn reads only what is new. **The full cache on every device**, the Intel GPU included; the window is a choice. All checked against a real b10621 build with Gemma 4's own template. **Vulkan for the Arc is not built**: it waits on the user's `llama-bench` numbers, SYCL against Vulkan | `quirky-pascal-7kx1wo` | #233 | merged |
| 31 | 09-29 | **Voice Box, phase 1a: whose turn it is on each card.** VibeVoice 7B judged feasible and a revised spec settled with the user (`docs/23-voice-box.md`, beside the user's intent document): the 7B from community mirrors, a selectable card, a request goes **next** on its card and never cuts the running job off, all VibeVoice first, one render per card, **warm between requests** and evicted by an image job, WanGP or the LLM, the image model **returned** (RAM or recipe, never disk), **never the pagefile**. Built: `mc_turns.py` (per-card turns; a guest is `resident_bytes`/`rendering`/`evict`, registered from outside so voice modules never import it; the **image gate** in `before_process` on txt2img and img2img via a new always-on `scripts/model_chain_turns.py`; Forge's `queue_lock` taken only when free; parking by `OPT_IMAGE_RETURN` — Automatic, From system RAM, From its recipe; `OPT_KEEP_WARM`; 4 GB RAM and 1 GB VRAM margins; a held job loads its own model and the gate joins an in-flight warm-up); `mc_memory.park_image_model`/`drop_image_model`; broker family `voice` (first victim of image and LLM, never a beneficiary); `mc_wangp.lease_*` against Mini Paint's lease (flush **asynchronous**); LLM turns wait for a render (`llm_wait_reason`, with the host-job exception) and a warm guest leaves when the LLM needs room. **Fixed on the way:** `mc_broker.cuda_execution` reduces a UUID to its hex, because nvidia-smi's `GPU-…` (the LLM's setup) and torch's digits (the image side) made one card two. No guest registers yet: phase 1b brings the VibeVoice runtime. Mini Paint NEO's half (lease, executor gate, bridge **1.12.0** `hold`/`resume`/`flush`; 1.11.0 went to its #110) built by a subagent and reviewed | `vigilant-meitner-5038fj` (both repositories) | #234; Mini Paint NEO #111 | open |

Branch names above omit the `claude/` prefix. Test-suite size grew roughly 1,577 → 5,424 across handoffs 01–27 and stood at 6,948 passed, 13 skipped at handoff 28, 7,102 passed, 13 skipped at handoff 29, 7,202 passed, 13 skipped at handoff 30 and 7,308 passed, 13 skipped at handoff 31; a large drop is a signal something is wrong.

---

## Standing conventions

These recur across the handoffs and the user has restated them repeatedly.

- **Develop on the designated branch.** Commit with clear messages, push when complete.
- **Do not open a pull request unless explicitly asked.** The user opens them from the Claude Code UI.
- **No model identifier** in commit messages, PR text, code comments, or anything else pushed to the repository.
- **Run the full suite** (`python3 -m pytest tests/ -q`), not just the file you touched. Ordering-dependent failures exist; a test passing alone may fail in the suite.
- **Mutation-check new tests.** A test asserting a new invariant must fail when the change is reverted. Several tests in this project's history passed for the wrong reason until checked this way.
- **When a behaviour change invalidates a test's setup** (not its intent), translate the setup rather than deleting the test.
- **Do not edit files while pytest is running** — it invalidates the run.
- **Update `README.md`** when declared behaviour changes.
