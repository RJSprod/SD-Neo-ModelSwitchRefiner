# Resource-scoped image + LLM concurrency — implementation notes

Companion to the *Resource-Scoped Image + LLM Concurrency with VRAM /
System-RAM Residency* design intent (27 August 2026). That document states what
the behaviour has to be; this one records the choices made against it, the two
places it was deliberately narrowed, and the handful of things that would
otherwise be rediscovered the hard way.

Section numbers below are the design intent's. `T` numbers are its acceptance
matrix, and every one of them has a test of the same name in
`tests/test_resource_scope.py`.


## 1. What this is, in one sentence

Every decision about who has to wait and what has to move is now scoped to the
physical thing the decision can actually change.

That sentence is the whole architecture. There is no new scheduler, no resource
manager, and no central authority: the same functions make the same decisions
they made before, and each of them now carries a card or a memory pool through
its arithmetic instead of assuming there is only one of each.


## 2. §4 — two vocabularies, because there are two questions

`ExecutionDomain` answers "which processor is this request materially using?"
`MemoryDomain` answers "if this residency goes away, which pool gains room?"
They are separate objects because they have separate answers, and Mixed
Conservative is the case that proves it: it executes on a CUDA card and keeps
every weight in system RAM. It competes with an image job on that card for the
*processor*, it competes with a warm checkpoint for *host RAM*, and it is not a
source of that card's VRAM for anybody. A single "on GPU" flag could not have
said that, which is why `Config.on_gpu` and `Config.uses_cuda_compute` were
already separate and why `execution_domain()` reads the second one.

`ExecutionDomain` has three states, not two:

| state | means | waits for an image job on the image card? |
| --- | --- | --- |
| `CPU_EXECUTION` | no CUDA device named | no |
| `cuda_execution(n)` | a resolved physical card | only when `n` is that card |
| `UNKNOWN_CUDA_EXECUTION` | CUDA, card unresolvable | yes |

The third row is not a spelling of either of the others, and keeping it apart
is invariant I-10. Both used to be "no card index", which meant one nullable
integer carried two states needing opposite decisions.

The asymmetry between the last two rows is deliberate and runs through
everything below: **an unknown card is conservative about execution and refuses
to be a reclaim victim.** False serialisation costs some throughput. False
reclaim destroys a prompt cache and can free zero bytes where they were wanted.


## 2a. The correction that came from a live machine

The first version of this compared card indices, and that was wrong in a way no
test in it could have caught, because every fixture stated that the two sides
agreed.

**A card index only means something inside the namespace that issued it.**
`nvidia-smi` numbers by PCI bus and is what `mc_llm_setup` records at setup
(`GpuInfo.physical_index` → `state["gpu_index"]`). The CUDA runtime numbers by
`CUDA_DEVICE_ORDER`, which defaults to *fastest first*. And a process launched
with `CUDA_VISIBLE_DEVICES` numbers from zero over whatever it can see. Three
schemes, and no reason for any two of them to agree.

From the machine that reported it: the language model was configured for
physical card 0, named "NVIDIA GeForce RTX 5090"; Forge reported device 0 with
24.0 GB total, which is a 3090. Both were "card 0". One equality, four
consequences, all in one log:

```
Waiting for image generation on GPU 0…
re-placing llama-server — it holds 19.3 GB where the active plan leaves 4.1 GB
released 13.1 GB of image VRAM on GPU 0 for Qwen 3.8 27B …
5 layers on the GPU … 17.2 GB free
```

A wait that was not needed; the 3090's image plan resizing the 5090's server; a
checkpoint evicted from a card the model was not going to; and a placement sized
against the wrong GPU's free VRAM, because `device_free_vram_bytes` was handed
an nvidia-smi index and passed it to `torch.device("cuda", …)`.

Two fixes, and they are separate:

**`image_device_index()` now answers in the physical namespace.** It translates
Forge's ordinal by UUID against `detect_gpus()`, so it is comparable with
`gpu_index` — which is what every caller already assumed it was.
`image_torch_ordinal()` keeps the old meaning for the one thing that needs it,
indexing `torch.cuda`. `device_free_vram_bytes(physical)` translates the other
way, and falls back to nvidia-smi for a card this process cannot address at all
— which is a real state, not a failure: Forge pinned to one GPU genuinely
cannot query the other through torch.

**Identity beats the index where the index cannot be translated.** An
`ExecutionDomain` carries a UUID and a card name, and `conflicts_with` climbs a
ladder: both UUIDs known settles it outright; two *different* model names settle
it in the negative (a negative only — two matching names may be two identical
cards); otherwise the indices, which is right whenever they share a namespace
and is what every single-GPU installation has always done.

The single-GPU path is deliberately untouched by all of it: one visible CUDA
device resolves to ordinal 0 without consulting anything, because there is only
one ordinal it could be.

What this cost in trust is worth recording. The acceptance matrix in §19 was
written against fixtures that *stipulated* `image_device_index() → 0` and
`gpu_index = 1`, so the suite could not have found this and did not.
`TestTwoNamespacesForOneSetOfCards` now builds the machine instead of asserting
its conclusion, and `test_end_to_end_through_the_real_translation` runs with
nothing about the card stubbed.


## 3. §7.1 — `ANY_CARD`, and why a sentinel rather than `None`

A card filter has three states — every card, one card, the card nobody could
name — and one nullable integer carries two. So `residencies(family)` means no
filter, `residencies(family, card=3)` means card three, and
`residencies(family, card=None)` means the residency whose card is unknown.
Overloading `None` for the first and the third is exactly how a query for "LLM
bytes on the image card" comes back with the other card's nineteen gigabytes.

`Residency.card` is recorded at declaration, from the configuration the process
was *launched* with rather than the one its role currently names — those differ
for as long as it takes a reconfigured role to restart, and a reclaim aimed at a
running server has to be judged against where the server is.


## 4. §7.3, §7.4 — two locks on one door

`RuntimeRegistry.release(card=N)` excludes every runtime whose reclaimable VRAM
is not on `N`, and `Runtime.on_card` refuses a targeted release for a card it is
not on. Both, because they fail differently: the filter is the policy and the
refusal is what survives a direct caller, a future regression in the filter, or
a stand-in that never learned about cards.

`_ask` in `mc_broker` offers the `card=` keyword and handles its refusal, and
what it does next depends on the family:

- the **image** family has one card by construction, so a card-blind answer *is*
  the card-scoped answer and the unfiltered call is used;
- the **LLM** family can hold two cards at once, so a card-blind answer is a
  machine-wide answer wearing one card's label. It comes back as "nothing
  eligible" instead, which costs a reclaim that did not happen and never costs
  the wrong process.


## 5. §16.1 — the one place a guess is allowed, and it is not a card

An unreadable image-card index means two different things and the difference is
how many cards the machine has.

On one card, everything is on it: the unfiltered answer is the card-local answer
and refusing to reclaim would break every single-GPU installation that works
today. On two, residency cannot be attributed at all, and stopping a server
chosen by guesswork can free every byte it holds and leave the shortfall exactly
where it was.

So `_reclaim_scope` asks `cuda_device_count()` — which is the *only* thing that
count is used for — and declines cross-family reclaim on a multi-card machine
whose image card cannot be named, saying so rather than doing something
plausible. A guessed GPU is not a reclaim target; an unfiltered request is not a
guessed GPU.


## 6. §8.3 — the plan baseline moved onto the runtime

`mc_plan._placed_for` was one module-level value, which was right for as long as
there could be one llama-server. With two, whichever started last overwrote the
other's, so a role on a second card inherited a boundary it had never been
evaluated against and was re-placed for a plan that does not describe its card.

`Runtime._placed_for` is now the answer to "which image-plan boundary has *this*
server been reconciled against". The module-level value is kept in step — the
panel reads it, and a single-server installation should see exactly what it saw
before — but nothing decides anything from it any more. A different-card runtime
records nothing at all, so nothing can later "move" and force a GGUF re-read for
a plan about another card (T16).

`_allowance` returns `-1` for both "no plan" and "the plan does not describe
this runtime's card". One value rather than two sentinels, because downstream
they mean the same thing: there is no plan-derived ceiling here.


## 7. §10 — host RAM, without a second cache

The repository already uses system RAM as a residency tier: `mc_memory` keeps
checkpoints, text encoders and VAEs warm there so a stage switch is a pointer
swap, and llama.cpp puts its weights there for every processor and Mixed
Conservative placement. Those are one physical pool. Planning them separately
fails in the same way mixing two cards' VRAM fails.

`mc_memory` gained exactly three functions —`warm_ram_bytes`,
`reclaimable_warm_ram_bytes`, `release_warm_ram` — and kept every decision about
what a cache entry is, which one is protected, and what "safe to drop" means.
The broker cannot reach a model object, because a broker that could reach one
would eventually be written as though it understood one.

Three things about `admit_host_ram` are load-bearing:

**If it already fits, nothing moves.** Sharing a pool is not a conflict
(invariant I-5). Two processor-resident servers and a warm Stage 2 checkpoint
that all fit above the reserve should all stay where they are; a scheduler that
emptied the cache because somebody else also uses RAM would be evict-on-switch
wearing a scheduler's clothes.

**The measurement wins.** A cache that reports fourteen gigabytes released has
said what it stopped referencing, not what the operating system has made
available again, so available RAM is re-read and that is what decides (T39).

**Stopping another server is not done there.** It is a stronger action than
dropping a cache entry, it depends on the user's role-sharing setting, and the
runtime layer owns both. `admit_host_ram` returns a shortfall and
`RuntimeRegistry._can_coexist_in_ram` decides.

What is *not* modelled: mmap'd GGUF pages. A full-GPU server's file may be warm
in the page cache and that warmth is the operating system's to reclaim, so
`host_ram_demand` returns zero for it (T41) rather than reserving a file's worth
of host memory against image work that is perfectly safe.


## 8. §17.9 — the clearest thing cross-domain awareness buys

Moving a language model from VRAM to system RAM solves a VRAM shortage by
creating a host-RAM demand of roughly the model's size. On a machine with room
that is the right trade: the server keeps answering, more slowly, and its prompt
cache survives. On a machine near its host floor it is not a trade at all — it
swaps a problem the driver can spill around for one that pages the whole
desktop.

So `_restart_in_system_ram` checks the destination first, and stops the server
instead when the destination cannot take it. The file pages stay soft-warm in
the OS cache for the next start, which is the better half of a bad choice.


## 9. §14 — the UI could not say two things

`mc_broker.active()` answers "which workload holds the broker's lock", which was
the same question as "what is running" only for as long as an image generation
and an LLM turn could not both be running. `activities()` answers the second: it
unions the LLM workloads with an image activity *derived* from `shared.state`,
because Forge starts its own generations and takes no lock of ours.

The global LLM workload lock is unchanged (non-goal N2). What changed is who has
to wait for it from outside.


## 10. What was deliberately not done

- **LLM-to-LLM requests still serialise.** Non-goal N2. Two servers may be
  resident at once and their requests still take turns, which keeps the scope of
  this change to image-versus-LLM concurrency.
- **No GPU-index-to-UUID migration.** Non-goal N8. The UUID configuration
  remains identity validation, not the scheduling key.
- **No page-cache management.** Non-goal N5. Nothing pins, flushes, or counts
  file-backed pages.
- **`request_vram` with no card still behaves as it always did.** Every existing
  caller that cannot name a card gets the machine-wide path, so nothing that
  worked before this change works differently after it on one card.


## 11. Where the interesting tests are

`tests/test_resource_scope.py` is the acceptance matrix, grouped as §19 groups
it, ending with the two end-to-end statements: `TestTheReportedMachine` (§19.10,
a 3090 generation and a warm 5090 server) and `TestTheHostRamRegression`
(§19.11, a warm Stage 2 checkpoint and a RAM-backed language model).

Two of its doubles are worth knowing about. `Server` defends its own card, like
the real `Runtime` does, so the registry-filter tests would pass against an
implementation with no filter at all — which is why
`test_the_registry_alone_excludes_a_runtime_that_does_not_defend_itself` uses
one that does not. And `_Unparseable` is a CUDA installation whose index is not
a number, which is the state that must not be read as "the image card,
probably".
