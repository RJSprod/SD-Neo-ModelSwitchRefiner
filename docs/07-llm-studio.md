# LLM Studio and cross-workload residency — implementation notes

Companion to the *Model Switch + LLM Studio for Forge Neo* design intent
(18 August 2026). That document is deliberately not an implementation recipe;
this one records the choices made against it, and — more usefully to whoever
picks this up next — the ones that were made *differently* and why.

Section numbers below are the design intent's.


## 1. What was brought across, and how

`prompt_master/` is a copy of `src/prompt_master/` from
[LTX_Video_Prompt_Claude](https://github.com/RJSprod/LTX_Video_Prompt_Claude)
at commit `eb80c60`, with two directories removed and **no edits at all**:

- `ui/` — the PySide6 window, chat page, MiniMax page, character editor, model
  chooser and setup wizard;
- `app.py` — the standalone Qt entry point.

Everything else is byte-identical, which is the point (§17): the prompt engine,
its vendored upstream, the MiniMax enhancer, the chat stores, the llama.cpp
client and the provisioning pipeline still `diff -r` clean against their
sources, and the parity tooling those trees carry still describes them
accurately. Updating means re-copying and re-deleting, not merging.

Two consequences are recorded in `prompt_master/VENDORED_FROM.txt` and repeated
here because they will otherwise be rediscovered the hard way:

- `core/paths.py::application_dir()` looks for `app.py` three directories up and
  will not find one. It is left exactly as upstream wrote it so the file still
  diffs clean; `mc_llm_paths.py` constructs `AppPaths` against a Forge-chosen
  root instead of letting it discover one.
- `setup_cli.py` is kept. It is the console half of the provisioning pipeline,
  it never needed Qt, and it remains a working way to provision a runtime when
  the Gradio panel cannot — a headless install, or a first run before the WebUI
  starts.

The Qt workers were not vendored, because they *are* presentation. Their
orchestration — the order of the passes, what is emitted when — was reproduced
in `mc_llm_sessions.py` as generators. The sequence of events is deliberately
the sequence the Qt signals had, because that sequence is the product: Prompt
Studio still emits its positive and negative separately (§4.2 forbids one merged
blob), MiniMax still emits its caption before its prompt (§4.4), Conversation
still emits one stream of reply text.


## 2. Where the design intent was followed exactly

| § | Requirement | Where |
| --- | --- | --- |
| 4.1 | One top-level Forge tab, compact mode selector, three dedicated workspaces | `mc_llm_studio.py` |
| 4.2–4.4 | Three modes that share a runtime and share nothing else | three `mc_llm_*_panel.py` modules |
| 4.5 | Output first, then session, composer, contextual controls, advanced | panel layouts + `style.css` |
| 5 | Extension-owned ids, scoped CSS, no Gradio-generated selectors, no hard-coded colours | `mc_llm_ui.py`, `style.css`, asserted in `tests/test_llm_panels.py` |
| 6 | llama.cpp kept as an isolated subprocess; GPU / partial / mixed / CPU all preserved | `mc_llm_runtime.py` over the vendored `LlamaProcess` |
| 7–10 | Hybrid and Exclusive modes, residency ranks, demote-only-under-pressure | `mc_broker.py` |
| 11–12 | Three separate budgets; per-model capacity estimation with calibration | `mc_llm_context.py`, `mc_gguf.py` |
| 13 | Graceful degradation with every reduction reported | `mc_llm_runtime.negotiate` |
| 14 | Concise status, detail in a collapsible view, "why" answerable | `mc_llm_studio._runtime_line` / `_residency_html` |
| 16 | Shared preferences separate from three independent histories | `mc_llm_state.py` + the vendored chat stores |


## 3. Where it was followed differently, and why

### 3.1 The workload lock does not span a generation

§15 asks for image and LLM execution to be serialised through "a shared workload
lock or equivalent coordination mechanism". A literal reading — hold the lock
from `before_process` to `postprocess` — is unsafe on this host: `postprocess`
is not called from a `finally`, so a generation that raises leaves the lock held
and every later LLM request blocked until the WebUI restarts. That failure is
much worse than the race it prevents.

The equivalent mechanism used instead is asymmetric:

- the LLM refuses to *start* a turn while `shared.state` says the host has a job
  running, and re-checks after taking the lock, because the check and the
  acquisition are not atomic;
- an image generation *waits*, once and with a bound, at the top of
  `before_process` for an LLM turn already in flight — and then proceeds holding
  nothing.

`shared.state` is maintained by the host in its own try/finally, so reading it
cannot leak. The residual window is the few microseconds between an LLM's last
check and its first token. What happens if it is hit is that the two overlap for
one completion: both are on the card, both are slower, neither is corrupted —
because co-residency was always allowed and it was only the *timing* that was
meant to be tidy.

### 3.2 The broker never asks a family to make room for itself

§9's eviction ranking is implemented for cross-workload decisions only. Image-on-
image eviction stays entirely in `mc_memory`, which has the cache bookkeeping and
the Forge entry points to do it properly; the broker would only be guessing at
both. `_victim_order` therefore never returns the asking family.

### 3.3 An image generation always outranks an idle LLM

§13's three policies govern the LLM's ambitions, not whether images can run.
Under every policy — including **LLM priority** — a foreground image pass can
reclaim an idle llama-server's VRAM, because §18's regression requirement
("ordinary txt2img/img2img remains functional") outranks a placement preference.

### 3.4 The warm tier for the LLM is the OS page cache

§7.2 asks that the intent of a warm layer survive where the mechanism differs,
and for llama.cpp it differs completely: there is no "move these weights to
system RAM" call to make. What there is instead is `mmap` — after one load the
GGUF's pages are resident in system RAM, and stopping the server does not evict
them, so a restart reads at memory bandwidth rather than from disk.

Nothing has to be done to get this except *not* do the one thing that would lose
it, which is to thrash RAM hard enough that the kernel drops the pages. That is
why the default release mode stops the server rather than reloading it
elsewhere. The alternative — restarting with `--n-gpu-layers 0` — is offered as
a setting for people who would rather spend the RAM than the reload.

### 3.5 Reduction before displacement, under Adaptive

§13 asks Adaptive to "demote whichever residency creates the least disruption".
The order chosen is: lower the LLM's context to its floor first, *then* ask the
image side for room. A context nobody is using is cheaper to give up than a
model somebody is about to use, and a 128k context sized automatically to fill
free VRAM is frequently exactly that.

### 3.6 Gradio 4.40, not newer

§5 says to target the components the host actually has. Two visible
consequences: `gr.Chatbot` is used with the `[user, bot]` pair value rather than
the message-shaped one, and grouped style options are flattened with the group
kept as a label prefix, because Gradio 4's `Dropdown` has no optgroup.


## 4. The estimator

The arithmetic is:

```
kv bytes per token = blocks × kv heads × (key width + value width) × element size
```

with each of those read from the GGUF header (`mc_gguf.py`), and both cache
halves sized separately because llama.cpp quantises them separately. This is
what makes the estimate model-specific in the way §12 requires: a grouped-query
model with 8 KV heads costs a quarter of what a 32-head model of the same width
costs, and no constant is right for both.

Weights and cache are close to exact. The compute buffer is not — it depends on
the batch size, the backend, and the graph llama.cpp plans — so it starts as a
coarse allowance and is then *replaced by measurement*. What is recorded is the
**overhead**: observed total minus the weights and cache the arithmetic already
accounts for. Storing the total would be useless the moment the context changed,
because the cache term scales with context and the overhead does not.

`Estimate.calibrated` is what lets the panel say "calibrated from a real load"
rather than "estimated", which §12 asks for explicitly.

One thing to preserve if this is refactored: **the estimator preview must not
reclaim.** `negotiate(reclaim=False)` exists because the panel is rendered when
the tab is built and every time the accordion opens, and a preview that evicted
a checkpoint to populate a table would be a worse bug than any it was drawing
attention to.


## 5. Layering, and the one hook

```
mc_llm_*_panel  ->  mc_llm_sessions  ->  mc_llm_runtime  ->  mc_broker
                                              |                  |
                                        mc_llm_context      mc_memory
                                              |
                                          mc_gguf
```

`mc_memory` does not import `mc_broker`. The image half stays importable,
testable and correct on an installation that never loads the LLM half — which is
also how §18's regression requirements are met by construction rather than by
care. What `mc_memory` has instead is one optional callback,
`set_foreign_reclaim`, installed by `mc_broker` at import and called only when
Forge's own eviction has fallen short. It never decides *whether* another
workload should give ground.

Two details of that seam are easy to get wrong and are worth stating:

- **Units.** `mc_memory` passes a *deficit* (bytes still missing, reserve
  already included). `mc_broker.request_vram` takes a *requirement* and
  subtracts free VRAM itself. `_reclaim_for_image` converts between them. Handing
  the deficit straight over subtracts free VRAM twice, evicts something, and
  leaves the pass short anyway.
- **Lock ordering.** The two reclaim paths take their locks in opposite orders,
  so `Runtime.release` acquires its lock with a timeout rather than blocking.
  Giving up costs an eviction that did not happen — which the caller already
  handles as a shortfall — where a deadlock would cost the whole WebUI.


## 6. What was not built

Recorded so it is a decision rather than an oversight:

- **No cross-extension protocol.** §19 lists it as a non-goal for a first
  release. The broker's `register_reclaimer` is shaped like one and could become
  one, but nothing outside this extension can register today.
- **No simultaneous image and LLM inference.** §19 again, and §8: co-residency
  is not co-execution.
- **No replacement of Forge's memory manager.** Every image-weight movement in
  this work goes through `memory_management.free_memory`, which moves weights to
  their offload device rather than discarding them — so a checkpoint demoted for
  an LLM stays in `mc_memory`'s RAM cache and coming back is still a warm swap.
- **No provisioning wizard.** Choosing an existing GGUF and projector, and
  reusing an existing standalone install, are both in the panel. Downloading a
  pinned runtime is left to the vendored `setup_cli.py`, which already does it
  and is already tested upstream.
- **No per-message affordances in Conversation.** Gradio 4's `Chatbot` has
  nowhere to hang them, so branching copies a thread from its last turn and
  "copy then delete back" replaces branching from a chosen point.


## 7. Tests

`tests/test_gguf.py`, `test_llm_context.py`, `test_broker.py`,
`test_llm_runtime.py`, `test_llm_studio.py`, `test_llm_panels.py` and
`test_cross_workload.py`. They run without a GPU, a model file or a WebUI.

Three are shaped by their failure modes rather than by their feature size:

- `test_broker.py` is mostly about evictions **not** happening, because
  "never unload merely because another workload started" is the invariant and
  every violation of it is silent;
- `test_cross_workload.py` exists only for the seams between the two halves,
  which is where the two bugs found during implementation both were;
- `test_llm_panels.py` builds every panel against the faked Gradio, because a
  UI this size is mostly wiring and wiring fails at build time or not at all.

What still needs real hardware: that a GGUF loads and answers, that a vision
projector actually sees an attached image, that co-residency of a real
checkpoint and a real LLM behaves as the estimator predicts, that calibration
converges on a real card, and that the tab survives a third-party theme.
