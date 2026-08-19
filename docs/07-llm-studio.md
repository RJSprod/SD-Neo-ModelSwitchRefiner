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
       |                                      |                  |
mc_llm_browse                           mc_llm_context      mc_memory
       |                                      |
mc_llm_files  <-  mc_llm_setup            mc_gguf
```

`mc_llm_files` imports neither Gradio nor the vendored package, which is what
lets `mc_llm_setup` use it and what lets it answer on an installation where
neither of those will import — the installation most likely to be typing a path
into the setup panel.

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
- **No full provisioning wizard.** `mc_llm_setup.py` covers the runtime — the
  part with no alternative — by three routes: detect one already in place,
  adopt one from anywhere on the machine, or download the pinned build where
  the manifest has one. Choosing a GGUF and a projector, and reusing an
  existing standalone install whole, are in the panel too. What is *not* here
  is downloading the pinned 16-27 GiB model, which `setup_cli.py` already does
  and is already tested upstream.

  This was originally left out entirely, and that was wrong: the panel could
  reach a state whose only recovery instruction was upstream's "Run Models and
  Hardware setup first" — a Qt wizard that does not exist in this extension. A
  dead end is worse than a missing feature, because the user cannot tell it is
  one. Two lessons worth keeping: an error message inherited from another front
  end may name a way out that this one does not have, and a two-step setup
  offered second-step-first is not a setup flow.
- **No per-message affordances in Conversation.** Gradio 4's `Chatbot` has
  nowhere to hang them, so branching copies a thread from its last turn and
  "copy then delete back" replaces branching from a chosen point.


## 6a. The containment rule, and what it costs

`AppPaths.contained` requires the runtime to resolve to a path inside the
install root, and `AppPaths.locate` deliberately does not require that of the
weights. Upstream's distinction, kept: the runtime is a program this extension
*starts*, the weights are a file it reads, and a state file that could point the
launcher at an executable anywhere on the disk is a different kind of object
from one that points at a model anywhere on the disk.

Three consequences fall out of it, all in `mc_llm_setup.py`:

- **Symlinks are out.** `contained()` calls `resolve()`, which follows them, so
  a symlink into the root resolves back outside it and is refused. Copying is
  the only honest way to satisfy the rule.
- **A release directory is copied whole**, because llama.cpp loads its shared
  libraries from beside the server and the executable alone would not start.
- **A system-packaged binary is copied alone**, because its directory is
  `/usr/bin` and copying that would be absurd. `_is_build_directory` is the test
  that separates the two cases — it looks for sibling `llama-*`/`ggml-*` files —
  and the single-file result is reported with its caveat rather than silently.

`MAX_ADOPT_BYTES` and `MAX_ADOPT_ENTRIES` are not a limit on how big a llama.cpp
release may be. They are the guard against being pointed at a folder that
*contains* a release rather than at the release.


## 6b. Paths typed into a text box

Three of the panel's controls ask for an absolute path, and for a while a text
box was the only way to supply one. That is a fine interface for somebody who
already has the path on a clipboard and a poor one for everybody else — and it
was worse than it looked, because *the obvious way to get a path onto a Windows
clipboard produces a path that names nothing*. Explorer's **Copy as path**
copies `"C:\models\thing.gguf"` **with the quotes**, and the resulting "there
is nothing at ..." names a path that reads as exactly right.

`mc_llm_files.py` is what stands between a paste and a `Path`. Everything it
undoes is something a real clipboard does:

| What arrives | Where from |
| --- | --- |
| `"C:\models\thing.gguf"` | Explorer's *Copy as path* |
| `file:///C:/models/thing%20one.gguf` | a file dragged onto the box |
| `%USERPROFILE%\models`, `$HOME/models` | a path written down in a note |
| a path with the line break still in it | a copy out of a chat window or a PDF |

It also answers two questions that used to be refusals:

- **A folder is an answer.** "My models path" is, to most people, the folder the
  models are in. Given that folder, the honest replies are "here is the one
  model in it" and "it holds these six — which one?". Refusing both with "there
  is no model file at ..." sends somebody back to the file manager to do work
  this can do.
- **A shard is corrected.** llama.cpp is handed `-00001-of-00003.gguf` and finds
  the other two itself; handed the third it loads a third of a model and fails
  oddly. Picking the wrong shard out of a folder listing is a reasonable
  mistake, so it is fixed and reported.

Everything the module works out is *reported and written back into the box* —
the panel's four outputs from **Use this model** exist so that a user who
pasted one thing and got another can see which. A silent correction would be
the same bug in a nicer coat.

`mc_llm_browse.py` is the other half: a picker, so the paths need not be pasted
at all. It renders the listing **server-side** rather than opening a native
dialog, and that is not a stylistic preference — a `tkinter` dialog would open
on the machine running the WebUI, which is either somebody's desktop (fine) or
a headless box where the dialog cannot be seen or dismissed and the worker
thread waits forever (not fine). One picker is built per box because Gradio
wires outputs statically and a shared one cannot know at build time which box
it will fill. Navigation binds to `input` where the host has it, because a
dropdown whose choices the handler replaces fires `change` on the replacement
and walks a folder deeper on every click.

## 6c. Panels that draw failure instead of raising it

`_estimator_html`, `_residency_html` and `_runtime_line` each wrap a body that
can raise. The estimator's wrapper is the load-bearing one: its HTML is *one of
four outputs of "Use this model"*, and Gradio discards every output of a handler
that raises. So a model could be recorded correctly, the panel show nothing at
all, and the only feedback be the word **Error** in a toast — which is
indistinguishable, from the user's side, from the model having been rejected.

The rule this settles on: a panel that cannot draw itself says which panel and
why, in the space where it would have drawn, and lets the rest of the click
stand.

The same class of confusion produced the memory radios' refresh on
`block.load`. Those controls are built once, when the WebUI starts, from
settings the Settings page can change afterwards — so the panel could show
*Hybrid* while the residency view directly under it said *Exclusive*. Both were
telling the truth about different moments.

## 7. Tests

`tests/test_gguf.py`, `test_llm_context.py`, `test_broker.py`,
`test_llm_runtime.py`, `test_llm_studio.py`, `test_llm_panels.py`,
`test_llm_files.py` and `test_cross_workload.py`. They run without a GPU, a
model file or a WebUI.

Three are shaped by their failure modes rather than by their feature size:

- `test_broker.py` is mostly about evictions **not** happening, because
  "never unload merely because another workload started" is the invariant and
  every violation of it is silent;
- `test_cross_workload.py` exists only for the seams between the two halves,
  which is where the two bugs found during implementation both were;
- `test_llm_panels.py` builds every panel against the faked Gradio, because a
  UI this size is mostly wiring and wiring fails at build time or not at all;
- `test_llm_files.py` is a list of real pastes rather than invented edge cases.
  Every string in it is something a clipboard actually produces, which is the
  only reason any of them is worth handling.

`test_llm_setup.py` has one fixture worth reading before adding to it: the
install root is a *subdirectory* of `tmp_path`, not `tmp_path` itself, so that a
test writing "a llama.cpp build elsewhere on the machine" is really writing it
outside the root. Rooting the install at `tmp_path` made every such build
contained and every containment assertion vacuous — which is how it was written
the first time, and all six containment tests passed while asserting nothing.

What still needs real hardware: that a GGUF loads and answers, that a vision
projector actually sees an attached image, that co-residency of a real
checkpoint and a real LLM behaves as the estimator predicts, that calibration
converges on a real card, and that the tab survives a third-party theme.
