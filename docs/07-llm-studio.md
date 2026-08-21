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

`krea/creativity/` is a second vendored package under the first, and is data
only — it carries its own provenance in
`prompt_master/krea/CREATIVITY_LIBRARY_SOURCE.txt` and is not from the
LTX_Video_Prompt_Claude tree at all. See `docs/08-krea-creative-mode.md`.

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
| 4.1 | One top-level Forge tab, compact mode selector, dedicated workspaces | `mc_llm_studio.py` |
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

The lock itself is not a `threading.RLock`, for a related reason. A workload
here is held across a generator, and a generator is not guaranteed to finish on
the thread that started it: Gradio runs a handler on a worker thread, and an
abandoned run — a cancelled one, a closed tab, one whose frame ended up in a
reference cycle because it raised — is finalised by the garbage collector, on
whichever thread happened to trigger the collection. `RLock.release` from that
thread raises `RuntimeError: cannot release un-acquired lock`, so the `finally`
that was giving the card back does not, and the lock stays held for the rest of
the session; because an `RLock` lets its owning thread re-enter, the damage is
invisible until a run lands on a different Gradio worker and waits for the GPU
forever. `mc_broker._JobLock` therefore records an owner for *reentrancy only* —
a chained generation is still one workload with two stages — and accepts the
release from anywhere, because `workload` releases exactly once per acquisition
and it is that pairing, not the thread it happens on, that keeps the count
honest.

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


## 4a. Not every block is the same block

The arithmetic above reads `blocks × kv heads × …` and that is how it was
first written — one scalar per model, multiplied by the block count. GGUF does
not promise that. The four attention keys may be **arrays with one entry per
block**, and llama.cpp writes them that way for every architecture whose blocks
differ from one another: hybrid attention/state-space models where only some
blocks attend at all, and interleaved local/global designs where the
sliding-window blocks are shaped differently from the full-attention ones.

Reading one of those as a scalar is not a rounding error. `int([8, 8, 0, 8])`
**raises**, and it raised inside a property — so choosing such a model put

> `int() argument must be a string, a bytes-like object or a real number, not 'list'`

in front of a user who had done nothing but pick a file, and put it there again
on every reply. The estimator panel showed it as a bare "Error" toast, because
its HTML is one of the outputs of *Use this model* (see 6c).

So `mc_gguf` reads these four keys as **tuples with one entry per block** —
`head_counts`, `head_counts_kv`, `key_lengths`, `value_lengths` — a scalar
being the one-entry case that applies to every block. The single-number
properties (`head_count_kv` and friends) are derived from the tuples for status
lines, rather than the tuples being derived from them. `_whole()` makes every
read total: a header that says something unexpected is a reason to estimate
coarsely and say so, never a reason for a property access to raise into a panel
that was only drawing itself.

`kv_bytes_per_token` then **sums per block** instead of multiplying. That is
not only about not crashing: multiplying the widest block by the block count
overstates a hybrid model's cache by a factor of two or more, which a user sees
as a context ceiling far below what their card can actually hold. A partial
offload is costed from the *end* of the model, because that is the end
llama.cpp offloads.

## 4b. One place the sampling numbers are written down

Conversation's temperature, top-p and reply-token defaults come from
`prompt_master.chat.characters` — `DEFAULT_TEMPERATURE`, `DEFAULT_TOP_P`,
`DEFAULT_MAX_REPLY_TOKENS` — and the seed's "draw a fresh one" sentinel comes
from `core.models.RANDOM_SEED`. The panel had its own literals, which happened
to hold the same numbers; the point is not the numbers but that a second copy
of them is how a front end quietly stops matching the engine it is a front end
for.

`ChatRequest`'s field defaults use `default_factory`, because a dataclass field
default is evaluated at *import* and nothing in the LLM half may import the
vendored package then (§18). A factory is evaluated per request, which is late
enough.

Below that, every control falls back twice: to the value the character was
saved with, and then to the vendored default. A cleared Gradio number box hands
the handler `None`, and the answer to that is the character's own setting —
never a crash, and never a literal invented in the panel.

Prompt Studio and MiniMax already worked this way and were left alone: their
per-pass temperatures live in `prompt_engine.motion`, `speech` and
`minimax.enhancer`, which carry upstream's numbers with upstream's reasons
beside them.

## 5. Layering, and the one hook

```
mc_llm_*_panel  ->  mc_llm_sessions  ->  mc_llm_runtime  ->  mc_broker
       |                                      |                  |
mc_llm_browse  ->  mc_llm_native        mc_llm_context      mc_memory
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
- **No response prefix in Conversation.** The standalone application can write
  the first words of a reply and have the model carry on from them
  (`prompt.prefix_instruction`, `Conversation.response_prefix`); the storage and
  the instruction are both vendored and unused here. It is a composer feature
  rather than a per-message one and was left for a later pass.


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

`mc_llm_browse.py` is the other half: **Browse**, so the paths need not be
pasted at all. It opens the operating system's own dialog
(`mc_llm_native.py`), because that is what the word means to everybody who
presses the button, and because reaching a models folder eight levels down a
drive is one keystroke there and eight clicks in anything drawn in a page.

The in-page listing is still there, as the fallback, because the native dialog
is not always the right thing to open — and this was originally built the other
way round, which was wrong:

- A WebUI started with `--listen` or `--share` is being looked at from another
  machine. A dialog opened by the server appears on the *server's* screen, and
  from the browser's side the button does nothing at all until it times out. So
  `mc_llm_native.available()` reads the host's own `cmd_opts` and declines
  before opening anything, and the in-page drawer opens instead **with the
  reason in it**. A Browse button that silently does nothing is the one outcome
  worth engineering around.
- A dialog runs in a **subprocess**, never in the Gradio worker thread. Tk is
  not thread-safe, a Tk that dies takes its process with it, and a dialog
  nobody dismisses would otherwise hold a worker forever. A child process can
  be killed on a timeout; a thread cannot.
- On Windows the route is **PowerShell and `System.Windows.Forms`** before
  `tkinter`, because the popular one-click Forge packages ship an embedded
  Python and an embedded Python has no `tkinter` — missing on precisely the
  installations most likely to want this. A route that reports itself missing
  falls through to the next; a route that timed out does not, because somebody
  who has already left one dialog open does not need a second.

One picker is built per box because Gradio wires outputs statically and a
shared one cannot know at build time which box it will fill. Navigation binds
to `input` where the host has it, because a dropdown whose choices the handler
replaces fires `change` on the replacement and walks a folder deeper on every
click.

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
  only reason any of them is worth handling;
- `test_gguf.py`'s per-block class is written from the format specification
  rather than from a model: the smallest model that would exercise it is 16 GB,
  and a synthetic six-block header exercises it in a millisecond.

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


## 8. The Conversation redesign (19 August 2026)

Six complaints, one shape. They are recorded together because four of them turn
out to be the same mistake made in four places: *the tab was laid out as though
it were a page rather than an application*.

### 8.1 The tab opened on the wrong mode

`_build` restored the mode selector from preferences and hard-coded
`visible=True` on Prompt Studio's column. A tab left on Conversation therefore
opened with the selector reading **Conversation** and Prompt Studio's panel
underneath it. Both controls were telling the truth about different things,
which reads — correctly — as the mode selector not working.

The fix is one expression (`visible=(value == initial)`), and the lesson is the
one §6c already records about the memory radios: **a control restored from
storage and a layout decided at build time are two states, and two states of
one fact will disagree.** Anything read out of preferences has to be read once
and used everywhere it is expressed.

### 8.2 Three columns, and the conversation in the middle of them

§4.3 asks for the conversation to receive most of the visual space, and the
first version answered it with proportions: a 1-part rail, a 4-part stage, a
2-part inspector. Proportions are not the same answer. A rail that is always
there is not secondary, it is *narrow*, and it costs the transcript a third of
the window permanently to hold controls that are looked at once a session.

So the two side columns are one drawer, hidden with `visible=False` rather than
made small. That distinction is the whole of it: Gradio does not render a
hidden column at all, so the stage is not "given more space", it is given the
window. Threads, the character and the persona are accordions inside it.

Choosing, editing and creating a character are also one section rather than
three places, because they are three things done to the same object: the
drop-down is who you are talking to, and the editor under it is that same
character, opened when it is being changed.

### 8.3 The transcript's height came from a number

`gr.Chatbot(height=560)` is an inline style, and an inline style is the end of
the argument — no stylesheet can respond to the window after it. So the panel
was 560px of transcript plus however much composer, action row and status the
mode happened to have, and whatever that added up to is what the page scrolled
to show. Scrolling a page to reach the box you type into is the one thing a
chat window must never make anybody do.

The component is now built with no height at all, and the height is decided in
two halves:

* `javascript/llm_studio.js` measures the distance from the top of the tab to
  the bottom of the viewport and publishes it as `--mc-llm-available`. This is
  the half that cannot be done in CSS: where the tab starts depends on the
  browser chrome, the host's header, how many rows the tab strip wrapped to and
  any theme in play.
* `style.css` makes the workspace that tall, makes the stage a flex column, and
  lets exactly one child grow. Everything else — the head, the action bar, the
  attachment row, the composer — is `flex: 0 0 auto` and measured by its
  contents.

Every `var(--mc-llm-available, …)` carries a pure-CSS fallback of the same
quantity, so the tab lays out correctly with the script absent and exactly with
it — which is §5's rule about JavaScript being enhancement, applied to layout.

Two escape hatches: below 900px the drawer becomes a full-width row above the
stage, and below 560px of *height* the page gets its scroll bar back, because a
transcript squeezed into nothing is not a transcript.

### 8.3a Two things the first attempt at that got wrong

Both were reported from a real browser and neither was visible from reading the
code, which is why `tests/test_llm_studio_js.py` now executes the arithmetic
under node.

**It measured the wrong element.** `--mc-llm-available` was measured from the
top of `#mc-llm-studio` and applied to the workspace — but the mode selector,
the model chooser and the status line sit in between, so the workspace was
given their height as well. That difference is exactly how far below the fold
the composer sat. It is measured from `#mc-llm-chat`, the workspace itself, and
published there.

**It fed itself.** The measurement was `innerHeight - getBoundingClientRect().top`,
and a rect's `top` falls as the page scrolls. So a scrolled page measured
*larger*; the workspace was then set taller; a taller workspace makes the page
taller; a taller page allows more scroll; the next measurement is larger again.
Over a dozen clicks a 756px workspace became 1485px, with the extra height
showing as blank space under the messages because Gradio's `.bubble-wrap` is
`height: 100%` over content that starts at the top.

The fix is to measure the element's position in the *document* —
`rect.top + scrollY` — which does not depend on the height being set, because
nothing above the workspace does either. Two guards sit behind it: the result is
clamped to the window height, so a measurement that has gone wrong costs a
workspace that is a little short rather than one that cannot be scrolled back
out of, and the workspace is `overflow: hidden` so a child that mis-measures
itself cannot make the page taller and start the loop again.

The scroll-invariance is the test worth keeping: same element, same place in the
document, two scroll positions, one answer.

### 8.3b The composer is sized first

"Fits the window" is not the same requirement as "the transcript is tall", and
when they conflict the composer wins — a thread too short to read is scrolled, a
composer below the fold cannot be used at all. So the stage states the priority
the only way a layout can: the head, the action bar, the image box and the
composer are `flex: 0 0 auto`, measured by their contents and never shrunk, and
the transcript is `flex: 1 1 0` with a deliberately small `min-height` and takes
whatever is left.

Three ceilings keep the fixed half honest, since all three sit between the
transcript and the bottom of the window: the composer is at most a third of the
stage (and its textarea stops at six lines rather than twelve), the action bar
at most 40% with its own scroll, and the image box at most 30%.

### 8.3c One gesture, both ways

A `Chatbot` bubble has one affordance — a click — so the click that opens the
action bar on a message is also the click that puts it away. Clicking a
*different* message moves the bar to it. The ✕ stays, because a bar left open
above the composer wants a way out that is not "find the message again", but it
is no longer the only one.

### 8.4 The upload box overlapped the buttons beside it

`gr.Image` was a column inside the composer row, sharing it with the message
box. A drop target has a minimum size of its own and does not shrink into the
gap left for a text box, so it drew over the send controls. It is now a row of
its own, hidden until **Attach** opens it — which also means the composer is
two controls rather than three for everybody not sending a picture, and that
the "this model has no vision projector" warning arrives when the box is
opened rather than when the reply fails.

### 8.5 Per-message actions, and the constraint that was mistaken for a wall

§6 recorded "no per-message affordances in Conversation" because Gradio 4's
`Chatbot` has nowhere to hang a `⋯`, and branching was reduced to "copy the
thread from its last turn, then delete back". That was the wrong trade. The
actions are the feature; the component only decides *where they are drawn*.

`Chatbot` has a `select` event. So a click nominates a message, and the actions
are one bar under the transcript applying to whichever message is nominated —
edit, regenerate, continue, send again from here, branch from here, delete,
delete from here, and the version pager. Everything the standalone application's
menu offers, in one row instead of sixty menus.

Three details are load-bearing:

- **The row/column a click reports is not a message index.** `Chatbot` takes
  `[user, bot]` pairs, so one exchange is one row holding two messages and two
  replies in a row are two rows with an empty left side. The map is built *with*
  the transcript, in one pass, and carried in a `gr.State`. Deriving it a second
  time somewhere else is how a click would come to name the message beside the
  one clicked.
- **Regenerating pages rather than replaces.** `history.Message` has kept its
  versions all along (`add_version`, `show`, `drop_version`) and the Gradio
  panel was throwing them away by deleting the reply and appending a new one.
  It now does what the standalone application does, which is what makes a
  regenerate reversible: the attempt that came back worse is undone with `◀`
  rather than by regenerating until luck returns.
- **The transcript is rebuilt, not patched.** Every action returns the whole
  view — rows, position map, selection, action bar — from one function, because
  they are one fact about one conversation and a handler that returned three of
  the four would leave the fourth describing a thread that is no longer on
  screen.

`Regenerate`, `Undo` and `Clear thread` are gone from the bottom of the panel.
The first is per-message now; the second is "delete from here" on your own
message; the third was a thread deletion with the file left behind.

### 8.6 Setup was a footnote to whichever chat was open

"Models, hardware and memory" was an accordion under the mode views. Two things
were wrong with that, and they pull in opposite directions.

Everything in it that is a **plain value** — context sizing, the buffer, the
context size, the two cache types, the residency mode, the policy, the release
behaviour, the folders — describes the installation rather than the click being
made. That is exactly the test this extension already applies to decide that a
control belongs on the Settings page (see the comments on `OPT_VRAM_RESERVE`
and the progress theme). So they are registered as Forge settings, the host
persists them in `config.json` with everything else, and they survive a restart
without this extension writing a line of storage code for them.

Everything in it that is **not** a plain value — the OS file dialogs, the
pinned-build download, the estimator, the residency table — cannot be drawn on
the Settings page at all: `ui_settings` builds one control per registered option
and there is no hook for a panel. Those are now **Setup**, a fourth mode in the
same selector as the other three.

`mc_llm_state.preferences()` is what keeps this from being a migration. It
reads three layers — the defaults, the preferences file, then the Settings page
for the five keys in `HOSTED` — so `mc_llm_runtime.config()` and the estimator
ask the same function the same question they always asked, and a headless
install with no `modules.shared` still answers from the file. `remember()`
writes both, because writing only the file for a key the Settings page is
authoritative for would make it look like it had worked and change nothing
anybody could observe.

One shape worth keeping if this is refactored: the option **default** for a
radio is a *label*, not a value (`label_for_context_mode`). What a Gradio radio
stores is the string it displayed, which is also why `mc_broker.resolve`
accepts either half of the pair.

### 8.7 What is left on the tab

A model chooser, `↻`, **Load**, **Unload**, and the status line.

The chooser is filled by `mc_llm_files.library()`, which walks the models folder
(`mc_llm_paths.models_root()`, its own setting because a models folder is very
often on another drive and shared with another front end) and drops the three
things nobody would choose to *run*: vision projectors, every shard of a split
model but the first, and anything below a depth cap so pointing it at a whole
drive costs a bounded walk. The **currently recorded model is added even when it
is outside that folder** — a model may be recorded from anywhere, and a chooser
showing nothing selected on an installation that works perfectly reads as the
model having been lost.

Choosing and loading are separate presses because they cost differently:
recording is a line in a file, loading is twenty gigabytes off a disk, and a
dropdown that loaded on every change would make scrolling the list an expensive
mistake. The chooser binds to `input` and not `change` for the reason §6b
already records for the picker's dropdowns — this code refills it on load and on
every rescan, and `change` fires on the refill.

The projector is deliberately not carried across a model switch or inferred from
the new model's folder. `model_choice`'s own reasoning stands: a projector has to
match the model it was made for and a file name does not prove that it does. One
sitting beside the new model is mentioned, and applying it is a press in Setup.

### 8.8 What the console is told

Every LLM run narrates itself to the WebUI console at INFO, through one wrapper
— `mc_llm_sessions._traced` — rather than through each mode logging for itself,
because what is worth saying is the same for all three: it started, what it is
waiting for or doing, how it ended, and how long it took. A run in progress says
so every five seconds with a character count, which is slow enough to read in a
terminal that is also carrying a generation's progress bar and quick enough that
a run which has stalled is visibly not moving. `mc_llm_runtime` adds the two
lifecycle lines around the one that was already there — starting llama-server
with the model, device, placement and context, and stopping it with the VRAM
released — and the tab logs Load, Unload and a model change.

**Nothing logged is content.** No prompt, no reply, no message, no character
name, no thread title. What is logged is the kind of run, the stage it reached,
sizes, elapsed times, and the model and device — which is the same class of
thing the image half already logs about checkpoints. Failures are logged with
their message, because the whole purpose of that message is to be read by the
person who has to fix it, and the vendored layers raise sentences about paths
and servers rather than about what was said.

`tests/test_llm_studio.py::TestWhatTheConsoleIsTold` asserts the second
paragraph, because the difference between a status line and a transcript of a
private conversation is one careless format string.

### 8.9 The top bar was four hundred pixels of chrome

Reported from a real install, and worth recording because none of it was a
layout bug — every element did exactly what it was told.

The status line said:

> Model: Q4_K_M · Device: NVIDIA GeForce RTX 3090 (24575 MiB, 23304 MiB free) ·
> Server: stopped

It was in a `gr.HTML` with no `scale`, in a `gr.Row` whose other child was a
`gr.Column(scale=4)`. A Column takes the width a Row's scale gives it and then
makes its contents fit *that*, however tall that turns out to be — so the
status was given the narrow remainder, wrapped to ten lines, and set the height
of the whole bar. The four mode radios wrapped to two rows underneath. Between
them the tab's chrome was taller than its conversation.

Three changes, in order of how much they were worth:

**The status is a chip, not a sentence.** One word — *Loaded*, *Unloaded*,
*Loading…*, *Not set up* — with a dot, `white-space: nowrap`, and a `max-width`
so it can never wrap the row it lives in whatever it is asked to say. The
sentence is not thrown away: it is the chip's `title`, it is the first line of
Setup's residency view, and every state change is already in the console with a
timestamp. `ui.state()` is the helper, and it escapes into an HTML *attribute*
now as well as into a body — a GGUF's `general.name` is still somebody else's
free text.

`_load_model` became a generator so the chip can say *Loading…* before the
load and *Loaded* after it. That is a state a status can only report by being
told, because at the moment it matters nothing has changed yet.

**The bar is flat.** One `gr.Row` with six controls and no nested Column, every
one of them `scale=0, min_width=0` except the model chooser, which is the only
control whose text is not known in advance and so is the only one allowed to
take the slack.

**It is sized as chrome.** `font-size: 0.9em`, tight button padding, and the
mode radios' labels given `white-space: nowrap` and small padding so four fit on
one line. 38px, measured, against roughly 400 before.

The composer's side buttons were the same class of mistake one level down:
Send, Stop and Attach stacked vertically were taller than the three-line message
box beside them, so *they* decided how tall the composer was. Stop and Attach
share a row now, and the stack stretches to the box's height rather than
setting it.

### 8.10 Following a reply, and the question that has to be asked first

The transcript is meant to do what every chat window does: stay at the end while
you are at the end, and hold your place while you are not.

The first version asked the question from inside the `MutationObserver` —

```js
const distance = target.scrollHeight - target.scrollTop - target.clientHeight;
if (distance <= FOLLOW_SLACK_PX) target.scrollTop = target.scrollHeight;
```

— and a `MutationObserver` runs *after* the new content is in the DOM. So a
reply taller than the slack made `distance` large for a reader who had been
pinned to the bottom a millisecond earlier, and the answer came back "no". There
is no reading of the DOM after the fact that answers "were you at the end?",
because the thing that moved is the definition of "the end".

So it is answered from `scroll` events, which fire only when the position
actually changes, and the observer does what the recorded answer says:

- **pinned** → `scrollTop = scrollHeight`, which the browser clamps to the real
  maximum, so the value is "the end" rather than a number that was right once;
- **not pinned, and `scrollTop` has collapsed to 0** → put it back. Nobody
  scrolled there: a full re-render empties the list for an instant and
  `scrollTop` is clamped against a `scrollHeight` that was briefly zero;
- **not pinned otherwise** → leave it exactly alone.

`FOLLOW_SLACK_PX` is 100 and not a number of this file's own, because that is
what Gradio's `ChatBot.svelte` uses in its own `beforeUpdate` check. Two
components disagreeing about whether you are at the bottom is worse than either
answer, and a scroller seen for the first time starts pinned, which is what
makes an opened thread show its newest message.

One honest note about how this was verified. Gradio 4.40 gets the algorithm
right for itself — its `beforeUpdate` captures pinned-ness before the DOM
changes, exactly as above — and in a browser harness that reproduces its update
cycle, it *masks* the old observer entirely: both the old and the new script
pass the end-to-end check. What does not pass is the observer driven directly,
which is what `tests/test_llm_studio_js.py::TestAnchoringTheTranscript` does by
capturing the callback and setting the scroller's numbers itself. Five of its
six cases fail against the old script. The behaviour now holds because this file
decides correctly on its own rather than because Gradio happens to decide first.

The per-message copy button went with it. It put an icon under every bubble in a
transcript whose whole job is to be read, to do something a selection and
Ctrl+C already do — and it was one more thing between two bubbles that are meant
to read as a conversation. There is no Copy in the action bar either: a server
cannot write to a clipboard, and a button that needs its own JavaScript to do
what selecting text already does is not worth the line.

## 9. The conversation that got slower with every message (19 August 2026)

A chat session with no image generation anywhere in it, in Exclusive mode, on a
24 GB card. Three replies, 35 s, 94 s and counting, and the console said why —
without anybody noticing what it was saying:

```
LLM a conversation reply — Starting llama-server…
starting llama-server — Q4_K_M …, 6 layers on the GPU, 8,192 token context
offload reduced to 6 of 30 layers on the GPU; the rest run from system RAM
…
LLM a conversation reply — Starting llama-server…
llama-server stopped — making way for a new placement
starting llama-server — Q4_K_M …, all layers on the GPU, 7,168 token context
llama-server ready — all layers on the GPU, 7,168 token context, 17.3 GB VRAM
…
LLM a conversation reply — Starting llama-server…
llama-server stopped — making way for a new placement, 17.3 GB of VRAM released
starting llama-server — Q4_K_M …, 2 layers on the GPU, 8,192 token context
```

Every message restarted the server. Every restart placed it worse than the last
one. The third reply was running two of thirty layers on a card that had been
holding all thirty a minute earlier.

### 9.1 A model negotiating against its own footprint

`negotiate` decides a placement from `mc_broker.free_vram_bytes()`, and
`Runtime.client` compares what it decided against the placement the running
server was started with. Different answer, different signature, restart. That
is the right shape when the question is asked once. It is a trap when it is
asked before every message, because **a running llama-server is the reason free
VRAM is low**. 17.3 GB of the card was this model; the negotiation was told
about the 5.5 GB beside it, concluded that the model would not fit, and placed
it in the gap it was about to leave — two layers, and the other twenty-eight in
system RAM. The next message asked again, from a card that now had 22 GB free
because the model was no longer on it, and swung back the other way.

Two costs, and the second is the larger one. Layers in system RAM are slow, and
that is visible in the log. What is not visible is that a restart throws away
llama.cpp's prompt cache: the server keeps the processed prefix of the previous
turn, so an ordinary reply only reads the new message. A replaced server reads
the *whole conversation* again before it writes a word — which is what
`generating, 1 characters in 67s` is a picture of.

So the negotiation is now told what it already owns. `already_ours` is added to
free VRAM in every fit calculation, and it is only ever correct because of what
the caller does next: every path that acts on the answer stops the running
server before it starts another one, so those bytes really are free by the time
anything is placed in them. With that one term, the same reading of the same
card gives the same answer twice running, which is all the stability the
comparison downstream ever needed.

Setup's estimator takes the same term, for the same reason and with the same
consequence if it does not: a table drawn while the model is loaded read the
card it is loaded on as a card with no room, and reported that the model
currently answering at 7,168 tokens could not be given a context at all.

### 9.2 Settings are honoured at once; arithmetic is not

The comparison itself changed shape too, because "the answer came out
different" and "the user asked for something different" had been the same test.
They are now two:

- **`_identity`** — the runtime, the model, the projector, the device, the
  offload and context the user chose, the KV types. Change any of these and the
  server is replaced immediately, as it must be.
- **`_worth_restarting`** — everything the card decided. More layers than the
  server is running, or a quarter more context, is worth a reload. Anything
  *less* is not worth anything: a running server holds its VRAM whether or not
  it is using all of it, so re-placing it smaller frees nothing anybody asked
  for. The image side has its own way of asking, and it is `release`, which
  stops the process outright.

A warm turn therefore costs a tuple comparison and one read of the model's
header, and asks the image side for nothing at all. That last part is a real
change in Exclusive mode, which used to sweep the image family before every
message: the sweep happens when the LLM is *placed*, which is when ownership of
the card actually changes hands. Re-sweeping before a message that is about to
be answered out of VRAM the LLM is already holding evicts a checkpoint to buy
nothing.

The recovery path is the same rule read the other way. A placement made while
the card was full is not permanent: `_outgrown` re-checks it before each
request with `reclaim=False` — a preview may never evict anything to answer a
question — and one restart buys back every layer once there is room for them.

### 9.3 Three things the console was saying wrongly

Fixed alongside, because all three cost time in reading the log above:

- **"LLM run abandoned"** after every completed reply. Gradio closes a
  handler's generator once it has consumed the last event, so `GeneratorExit`
  arrives a moment after the run says it is complete. Only a run that had not
  finished is abandoned now.
- **"Starting llama-server…"** before every message. True when every message
  restarted it. A warm turn says `Preparing…`.
- **"on NVIDIA GeForce RTX 3090 (24575 MiB, 23304 MiB free)"** beside a live
  placement decision. That parenthetical is part of the device name recorded
  during setup and is a snapshot of a moment months ago; sitting in the same
  sentence as the placement it appears to explain, it is the number a person
  debugging that placement will believe. The name is kept, the parenthetical is
  dropped, and the line says what was actually free when the decision was made.

### 9.4 The activity bar

The other half of the same report: with a minute between a message and its
reply, nothing on the page said the request was alive.

It is two pixels along the bottom edge of the status line that is already
there, so a reply starting moves nothing — not the transcript, not the
composer. It is indeterminate, because nothing on the server knows how many
tokens a reply will run to and a bar filling towards an invented number is a
worse answer than one that only claims the request is alive. `ui.working` is
`ui.notice` plus one class, so a theme that styles status lines styles this
one, and the bar is the only difference between them.

Which lines get it is a decision about truth rather than decoration. A busy
line is one a run is still working behind: "Starting…", "Waiting for image
generation…", "Replying…", and "Stopping…" — a stop asks the run to finish and
what is already streaming keeps arriving until it does. "Reply complete.",
"Cancelled." and every error are `ui.notice`, because a bar still sweeping
under a finished run says the opposite of the sentence printed beside it. The
top bar's state chip does the same thing where there is room for a dot and not
for a bar: `Loading…` pulses.

Two details are worth their lines. The bar is a `<span>` this extension emits
rather than a `::after` on the status line, for the reason the rest of the
stylesheet gives about the host's progress bar and `tests/test_progress.py`
then holds the whole file to — a pseudo-element may already be somewhere a
theme is drawing. And under `prefers-reduced-motion` it is reduced rather than
removed, for the reason the ooze theme is: it is the only thing on the page
saying the request is still alive, and a ninety-second request is exactly when
somebody needs to be told. The sweep becomes a still bar along the whole edge.

The elapsed count beside it is in `llm_studio.js`, and that is not an arbitrary
split. The server could have written the number; it could not have kept
writing it, because a status is repainted when the run yields and the whole
complaint was runs that yield nothing for a minute while a server loads and
reprocesses a prompt. A clock that stops during the wait it exists to measure
is worse than no clock. The bar is CSS and runs without the file; the number is
the one thing here that needs a second to be able to pass without a round trip.
Where the clock is *kept* is the part with a bug in it, and the part the tests
drive: a run's status line is replaced wholesale every time the run says
something new — "Starting…" and "Replying…" are separate elements, not one
element with two texts — so a start time stored on the line resets at each of
them and a ninety-second reply reads as five. It is kept on the component,
which survives the run, and cleared when the line stops being busy.

## 10. Four reports from the same afternoon (19 August 2026)

### 10.1 A CUDA library that was not the problem

> `[WinError 5] Access is denied: 'C:\Roots\Neo3\model_chain_llm\runtime\cublas64_12.dll'`

— on pressing **Use this runtime** after changing the device. The filename is
the reason this took a while to read: it names a CUDA library, so it reads as a
driver or a permissions problem, and it is neither. Windows will not rename or
delete a file that a process holds open, llama-server holds every DLL beside it
open for as long as it runs, and adopting a build replaces that whole folder —
`_copy_tree` renames the old `runtime/` out of the way, `extract_zips_atomic`
deletes it outright.

`_apply_runtime` already stopped the server. It stopped it *after* the copy, on
the reasoning that the running server holds the build it was started with — true,
and the exact reason the stop has to come first. Both handlers now stop before
they touch the folder, and both paths turn a `PermissionError` on the
replacement into a sentence naming the folder and saying to press Unload,
rather than a Windows error code pointing at cuBLAS.

### 10.2 The character controls that emptied into a blank gap

Open the drawer, open **Threads**, and the **Character** section below it loses
its contents — leaving the heading, a gap, and no scrollbar.

The drawer is a `gr.Column`, which Gradio renders as a flex column, inside a
workspace whose height is fixed by `--mc-llm-available`. Its sections are
therefore flex items with the default `flex-shrink: 1`, and the Threads
accordion is as tall as the thread history is long. When the total stopped
fitting, the sections underneath were *shrunk* rather than the drawer being
scrolled — and `overflow-y: auto` never fired, because from the drawer's point
of view everything fitted exactly.

```css
#mc-llm-studio .mc-llm-drawer > * { flex: 0 0 auto; }
```

The stage next door has carried the same rule since it was written — the
action bar, the attachment row and the composer are all `flex: 0 0 auto` there,
so that the transcript is the one thing that gives ground. The drawer was the
half of that idea that never got written down.

### 10.3 The Stop button that walked off the bottom of the screen

A Gradio `Textbox` grows from `lines` towards `max_lines` as text arrives.
MiniMax's output box was `lines=18, max_lines=44`, so it grew by twenty-six
lines *while a generation streamed into it* — carrying the composer, the
Enhance button and Stop down the page ahead of it, at the one moment somebody
wants to press Stop. Prompt Studio's two outputs were 16 and 40.

`max_lines == lines` on every box a generation writes into. A box that cannot
change size cannot move a button, and text longer than the box is read by
scrolling inside it, which `overflow-y: auto` in the stylesheet guarantees
whatever the component decides for itself.

### 10.4 "It says GPU only, but it is slow"

The restarts were gone — the log showed one start and then `Preparing…` on every
message after it, which is what section 9 was for. What remained was a fully
resident model generating at roughly a fifth of the rate a fully resident model
on that card generates at, with a prefill to match: fifty-one seconds to the
first character of a seven-thousand-token context that a 3090 should chew
through in a few.

Nothing this extension had written down could tell anybody why, and that is the
finding. `all layers on the GPU` is a *decision*. `18.1 GB VRAM` is a
measurement of free memory before and after, which cannot tell VRAM that is on
the card from VRAM the driver has quietly backed with system memory. Between
those two numbers there was no evidence at all about what llama.cpp actually
did — and llama.cpp writes it down, in its own log, in the folder nobody opens:

```
load_tensors: offloaded 31/31 layers to GPU
load_tensors:        CUDA0 model buffer size = 17000.00 MiB
load_tensors:   CPU_Mapped model buffer size =   300.00 MiB
```

So it is read back, from the byte the log ended at before this start, and
reported on the line after the placement. When a tenth or more of the weights
are in system RAM, that is a warning rather than a note, and it says where to
look — another process holding VRAM, or a driver spilling an allocation it
could not fit, which on Windows is a policy with a name and a setting.

The threshold is not zero on purpose: a full offload still leaves a
token-embedding buffer on the host for many models, and a warning that fires on
every load is a warning nobody reads.

One unknown was removed while the instrument was being fitted.
`--n-gpu-layers` was being passed the word `all`, which is this project's word:
llama.cpp's own argument is an integer, and every invocation of it in the wild
passes one. It is now the model's own block count plus one for the output
layer, or 999 when the header could not be read — deliberately far too big,
because llama.cpp clamps a count above the model's own while a guess that came
out too small would silently leave layers on the processor. A build that does
not understand `all` refuses to start rather than offloading less, so this was
never the cause of a slow reply; it is one fewer thing a slow reply could be.

## 11. The drawer that was drawn outside the box (19 August 2026)

Section 10.2 fixed one half of this and named the other half without seeing it.
The sections in the drawer were being shrunk to fit, so `flex: 0 0 auto` on the
drawer's children stopped that — and the next report was worse: expanding
**Character** made the whole drawer blank, with a screenful of nothing above a
stage sitting on the bottom edge of the workspace.

Which is the same bug seen from the other end. The workspace is a Row with a
fixed height and `overflow: hidden`. What decides how tall its two columns are
is `align-items`, and that is a property the host's own Row — or a theme over
it — is entitled to set. Aligned to an edge rather than stretched, both columns
are sized by their own contents: the stage becomes as tall as a status line, a
transcript and a composer, which is why it sinks to the bottom with a gap above
it; and the drawer becomes as tall as its contents, which once the Threads list
is open is taller than the row, so its top half is pushed out through the clip.
Nothing was hidden. It was drawn outside a box that does not scroll, and
stopping the sections from shrinking is exactly what let them grow far enough
to be pushed out.

Three declarations, and each one is a different half-truth removed:

- `align-items: stretch` on the workspace — both columns are the row's height,
  whatever anything upstream would rather they were;
- `height: 100%` on each column — the same statement made where a percentage
  can resolve against the row's own fixed height, so it does not depend on the
  first being honoured;
- `min-height: 0` on the drawer — a flex item's automatic minimum size is its
  content size, so without it the declared height is overridden from
  underneath and the `overflow-y: auto` that has been on the drawer all along
  never has anything to scroll.

The stage has carried `min-height: 0` since it was written. The drawer had
`min-width: 0` and never its opposite number, which is the kind of asymmetry
that survives every reading of the file and none of the resizes.

## 12. Two questions the console could not answer

### 12.1 Where the log is

`<LLM data directory>/logs/llama-server.log` — beside the runtime and the
models, not in the extension folder, because the extension folder is a git
checkout that an update overwrites. That was already true and written down in
exactly one place nobody had reason to open.

So it is printed, in full, on every start, and it is in Setup's residency panel
whether or not anything has gone wrong. One line per start, and starts are rare
now.

### 12.2 What llama.cpp reported

Section 10.4 added a line reporting llama.cpp's own load report. It never
appeared, and the reason is the same reason the report is worth having: what a
program says about itself and what a reader can see are two different things.
`/health` answers the moment the model is loaded, and what llama-server wrote
while loading is on the other side of its own output buffer — which, when the
output is a file rather than a console, is a block buffer on Windows rather
than a line one. Reading once, immediately, found an empty file on the platform
that most needed the answer.

It is now read for up to five seconds, stopping the moment there is something
to read, and a start that still has nothing says so with the path rather than
printing nothing at all: with silence after the placement line there is no way
to tell a report that said everything was fine from a report that was never
read.

### 12.3 "Nothing evictable was found"

True, and on its own misleading. It reads as though the card were full of
things this extension chose not to move; what it usually means is that the card
is full of something it cannot see at all. A user's log had the LLM short by
18.2 GB with 4.7 GB free, the image side reporting nothing resident and freeing
nothing when asked — because there was nothing of the image side to free. Some
other process had nineteen gigabytes of that card: another program, or a
llama-server left running by a WebUI that was killed rather than closed, which
holds its allocation for as long as it lives and is invisible to every check
here.

`unaccounted_bytes()` subtracts what each family admits to holding, and a
gigabyte for the driver and the desktop, from what the card says is in use.
When the remainder is real it is named — in the shortfall note, and in Setup's
residency panel, which is the panel somebody opens when a placement makes no
sense and is the one explanation no row in its table can ever show.

It also subtracts a second gigabyte when a llama-server of *ours* is up holding
nothing on the card. A server placed in system RAM reports nothing resident and
declares nothing, which is right — its weights are not there and the image side
must not come looking for them — but its process is on the card all the same,
and a CUDA context is hundreds of megabytes before a single weight is loaded.
That is what the driver's own gigabyte allows for, so a second CUDA process
gets a second allowance. Without it a user running the LLM entirely in system
RAM was told, on every roll, that 0.9 GB was held by "a llama-server left
running by a previous session" — which was their own, running on purpose, and
not something any amount of hunting through `nvidia-smi` would have fixed.
`stray_explanation()` therefore branches on whether we have a server up, and is
shared by the console note and the panel so the two cannot drift into two
accounts of one card.

### 12.4 A reason is a noun phrase

Every message built from a `reason` reads it as the subject of a sentence:
"X is short 2 GB", "freed 2 GB for X", "released 2 GB of image VRAM for X".
Half the callers passed a clause instead, and the same user's console read
`a Krea image generation follows is short 18.5 GB`. The reasons are noun
phrases now — "the image generation that follows a Krea roll", "the LLM
workload taking VRAM ownership" — and so is the fallback for a request that
did not say, which used to be the bare family key and produced "llm is short
2 GB".

## 13. The drawer, and a llama-server that would not start (19 August 2026)

### 13.1 It expanded to the right

"When it expands, it expands to the right off page, instead of expanding
vertically."

That sentence names the mechanism, and it is the third and last thing wrong
with the same eight lines of CSS. The drawer's rules arrange three sections in
a column, scroll them, and stop them being squashed — and never once say that
the sections are in a column. `display` and `flex-direction` on a host
container were left to Gradio, and laid out as a *row* instead, with the
children told not to shrink by §10.2, the three sections stand side by side at
their natural widths and walk straight off the right-hand edge of the page.

The stage next door has declared both since the day it was written. Every fix
to this drawer has been a property the stage already had:

| | stage | drawer |
| --- | --- | --- |
| `display: flex; flex-direction: column` | from the start | §13.1 |
| `min-height: 0` | from the start | §11 |
| `flex: 0 0 auto` on children | from the start | §10.2 |
| `height: 100%` / `align-items: stretch` | §11 | §11 |

A column that was written as "the drawer is the narrow one" and a column that
was written as "this is a flex layout and here is what each rule is for".

### 13.2 And it stays where it can be read

The drawer is a fixed-height scrolling column, so a section opened near the
bottom of it opens below the fold — the click landed on the heading, and the
heading was already visible, which is why no browser fixes this for you.

`llm_studio.js` brings it back: after the section has had a moment to open, the
drawer is scrolled by the smallest amount that shows it. Deliberately not
`scrollIntoView`, which scrolls every scrollable ancestor including the page,
and the page is the one thing this tab does not move. The rule is arithmetic —
already visible, taller than the drawer, or below the fold — so it is run in
the test harness rather than described.

### 13.3 The load report that was never in that format

§12.2 said the load report was missing because it had not been flushed yet.
That was a good guess and it was wrong, which the user's own
`llama-server.log` settled in one line:

```
common_init_result: fitting params to device memory ...
```

This is a 2025 build with its own memory fitter. It logs none of the three
lines the parser was written against — no `load_tensors:`, no per-buffer
accounting, no `offloaded N/M layers to GPU`. What it does log is the context
it settled on, what it saw free on each device, why a load failed, and, after
every request, how fast that request actually ran. All four are now read, and
the fixture in `tests/data/` is that build's own output, kept because a format
nobody can reproduce from memory is a format that quietly stops being parsed.

Two of them earn their place immediately. The context llama.cpp *settled on* is
not always the one it was asked for — 7,168 asked, 6,912 run — and a number
this extension reasons about while the server runs a different one is worth a
line. And the timings are the measurement every other number here is a proxy
for: on that one card, that one model and that one week, the same placement
reported by this extension as "all layers on the GPU" produced **106 tokens per
second** on the first start and **2.68** on a later one. Characters per second,
which is all this side can count, cannot tell those apart from a chattier
model. `llama.cpp measured 6.7 tokens/s` on the end of the run line can.

### 13.4 A card with room that refuses to give it out

The run where nothing worked at all:

```
device_info:
  - CUDA0 : NVIDIA GeForce RTX 3090 (24575 MiB, 23304 MiB free)
ggml_backend_cuda_buffer_type_alloc_buffer: allocating 18231.52 MiB on device 0:
    cudaMalloc failed: out of memory
llama_model_load: error loading model: unable to allocate CUDA0 buffer
srv llama_server: exiting due to model loading error
```

22.8 GB free, and a single 17.8 GB allocation refused. That is not a
contradiction, it is the difference between how much memory a driver has left
and how much of it it will hand out in one piece — and Windows is stricter
about it than any arithmetic in this module can model. Every check here passed;
the placement was sound; the server died anyway, and what reached the user was
"llama-server exited before becoming ready", which is true of every failed
start and useful for none of them.

Two changes, and the order matters:

- **Say what happened.** The reason is in the log, one line before the server
  goes. It is read and raised: *"llama-server could not fit on the card: it
  asked the driver for 17.8 GB in one piece and was refused (out of memory),
  with 22.8 GB reported free."* A sentence somebody can act on, in place of a
  sentence about a process exiting.
- **Then try again with less.** Nothing here could have predicted the refusal,
  so it is learned instead: up to two more attempts, each holding back three
  more gigabytes than the last, each one logged. Three gigabytes rather than
  one because a step that only trimmed the context would ask the driver for the
  same allocation it had just refused — the weights are what did not fit, and
  only fewer layers make that number smaller.

Only an out-of-memory failure is retried. A corrupt file, a missing projector
or a port already in use fails once, immediately, with its own sentence.

## 14. Free memory that was not free (19 August 2026)

The retry ladder from §13.4 worked exactly as designed and did not help:

```
allocating 18231.52 MiB on device 0: cudaMalloc failed: out of memory
allocating 15244.86 MiB on device 0: cudaMalloc failed: out of memory
allocating 10588.65 MiB on device 0: cudaMalloc failed: out of memory
```

Three placements, each smaller than the last, every one refused — on a card
whose own start-up line said `24575 MiB, 23304 MiB free`, sixteen times in one
session. A driver that will not hand out ten gigabytes of the twenty-two it
says it has is not being cautious about fragmentation. It does not have them.

### 14.1 Two questions, one number

`mc_memory.free_vram_bytes` is the host's own figure, and the host computes it
the way every ComfyUI derivative does:

```python
mem_free_cuda, _  = torch.cuda.mem_get_info(dev)
mem_free_torch    = mem_reserved - mem_active      # the allocator's cache
mem_free_total    = mem_free_cuda + mem_free_torch
```

That second term is right, and it is right *for the host*. PyTorch keeps the
blocks it has finished with, because keeping them is what makes the next
allocation fast, and it will reuse them before it asks the driver for anything.
Counting them as free is how a checkpoint knows it can load.

For another process they do not exist. llama-server cannot be handed a block
PyTorch is sitting on, and llama-server is the whole point of this half of the
extension: section 6's separate process, chosen deliberately, and paid for in
exactly this coin. Every placement decision in `mc_llm_runtime` was being made
against a number that included between twelve and nineteen gigabytes of memory
the model it was placing could never have — so the arithmetic said "all thirty
layers fit", the driver said "out of memory", and the extension's own reading
of the card agreed with the arithmetic right up until the process died.

It also explains the forty-fold speed spread in §13.3 without any new theory. A
build with its own memory fitter reads the *driver's* free memory, not the
host's, and quietly leaves expert tensors in system RAM to fit what it really
sees. Two starts, the same placement on paper, 106 tokens per second and 2.68.

### 14.2 The fix, and the part of it that is not arithmetic

`device_free_vram_bytes()` asks the driver — `torch.cuda.mem_get_info`, without
the allocator's cache added back — and every figure the LLM side places against
is now that one. Two functions because there are two questions: what can *this*
process spend, and what can *another* process be given. Answering the second
with the first is the whole of this bug, and the broker now picks by family:
an image pass gets the host's figure, an LLM request gets the driver's.

Reporting it honestly would have been enough to stop the crashes and not enough
to make anything work — a truthful four gigabytes places two layers of thirty
on the card and runs the rest from system RAM. So the cache is handed back
before a server is placed. `release_cached_vram()` is `soft_empty_cache`, which
unloads nothing and moves no model: what it gives up is the *empty* space
between the things that are loaded, which is worth doing exactly once, at the
one moment another process is about to be asked to fit in it.

Only on a path that is really going to start a server. A warm turn touches
nothing, and neither does the preview `_outgrown` runs before every request —
emptying an allocator to answer a question nobody asked would be a fine way to
turn this fix into the next regression.

## 15. The drawer, again, and for the last time

Three reports, one shape: sections side by side, sections cut off, a toggle
that would not close. Every fix so far has been a CSS property the stage next
door already had, and each one moved the problem rather than removing it,
because the thing being fixed was a layout the host was free to disagree with.

So the accordions are gone. The drawer is a chooser and three sections, exactly
one of which is in the layout:

```
[ Threads | Character | You ]     <- a radio, stuck to the top of the drawer
[ the chosen section              ]
```

What was asked for was "always vertically stacked, one open at a time, and the
open one takes the emphasis until I close the drawer". An accordion cannot
promise any of the three — it is laid out by the host, it opens independently
of its neighbours, and it is as tall as whatever is inside it. Three columns
and a radio promise all three by construction, in a drawer of fixed height
where two open at once would leave neither readable.

The toggle button now carries its own next action as its label: **☰ Threads &
character** when the drawer is shut, **✕ Close panel** when it is open. That is
not decoration either. The open/closed flag lives in a `gr.State` and the
visibility lives in the component, and nothing in Gradio keeps two such things
in step: a reload can leave the state saying "closed" underneath a drawer that
is on screen, and then the press that should close it opens it again — which is
exactly what "it opens but does not toggle close" looks like from outside. A
button that says which way it is about to go is one whose next press is
predictable, and one that says the wrong thing is a bug somebody can see and
report rather than a control that feels dead.

## 16. The context window was already rolling

Worth writing down because it is the kind of thing that gets asked twice.
`prompt.build` sizes a budget from the context, subtracts the system prompt and
the reply allowance, and keeps the newest messages that fit — oldest first out,
last message always in. A conversation longer than the window has never been
sent whole.

What it was sized against had drifted, though. The budget came from the
placement this extension *asked* for, and a build with its own fitter answers
with 6,912 tokens against a conversation trimmed to fit 7,168. It now prefers
the context llama.cpp reported, then the one that was asked for, then the
setting — most-informed first, which is the only order that is ever right.

## 17. Stop left the panel unable to do anything else

`stop.click(..., cancels=[running])` closes the running generator where it
stands, which is what makes a stop immediate. It is also why the generator
never reaches the yield that would have re-enabled the submit button and greyed
out Stop: there is no code left to run. So the run stopped, the partial output
stayed — both correct — and the panel sat there permanently busy with no way to
ask for anything else.

Whatever restores those controls has to be the stop handler, because it is the
only thing that still runs. All three panels now return the two button updates
along with the status line, and the click is wired to all three outputs, which
is the half that is easy to leave out and the half a test can catch.

## 18. Memory this extension does not own

The report that prompted this section is worth quoting, because it is the right
complaint: *"I should not have had to clear the system RAM from another app to
make things work."*

Some of that is fixable here and some of it is not, and the honest thing is to
be exact about which is which.

### 18.1 What was fixable, and is fixed

**The vision projector was being loaded for text-only turns.** llama.cpp
announces its own worst case for it — 1,285 MiB for this model's f16 projector
— and that memory comes out of the same budget as the weights. A gigabyte and a
third on a card that is already within two of its limit is not a rounding
error: unaccounted for, it is a gigabyte and a third llama.cpp has to find
somewhere, and where it finds it is by leaving part of the model in system RAM.

It is now loaded for a request that carries an image and for no other, and its
size is counted against the card when it is. The cost is one restart the first
time a picture is attached, which is the trade the right way round.

### 18.2 What is not fixable here, and is now said out loud

**Another application's memory.** There is no call this extension can make that
takes system RAM back from a program that is using it, and there is none that
takes VRAM back from a process it did not start. Section 12.3 already reports
VRAM on the card that neither family admits to holding; this adds the other
half. Before every start, free system RAM is compared against the size of the
model file — llama.cpp reads a GGUF through `mmap`, so the file goes through
the page cache whether the weights end up on the card or not — and a shortfall
is a warning naming both numbers. It is a warning and not a refusal: it is the
machine's memory, the user may know exactly what else is using it, and
llama.cpp will make an honest attempt either way. What is not acceptable is for
it to be slow for a reason nothing on screen mentions.

**llama.cpp's own fitter.** A 2025 build fits the model to what *it* sees free
and does not say what it decided. Told to put thirty layers on the card, it
will still move expert tensors into system RAM to make room for its context and
compute buffers, and the only external sign is the reply rate.

So the last thing this extension can do without reading anybody's log is
arithmetic on its own measurement: the placement says all thirty layers, the
weights are seventeen gigabytes, and the card's free memory fell by four. That
is now a warning naming both figures, and it works on every build of llama.cpp
because it reads nothing llama.cpp wrote.

### 18.3 What is on screen now

Setup's residency panel carries the four numbers that separate the cases, and
the console carries all of them at every start:

| what it says | what it means |
| --- | --- |
| VRAM in use by something this WebUI is not managing | another process has it; nothing here can reclaim it |
| only N GB of system RAM is free and the model is M GB | close something, or expect a slow load |
| the card took N GB where this placement needs M GB | the weights are not all on the card, whatever the placement says |
| Last reply: llama.cpp measured N tokens/s | the only number that is not a plan |

## 19. Conversation rebuilt as a messaging application (19 August 2026)

Every fix in sections 10, 11, 13 and 15 was a fix to the same thing: a drawer
beside a transcript, in a row, laid out by a host that is entitled to disagree
about how a row works. The sections stopped being squashed, then stopped being
drawn outside the box, then stopped expanding to the right, then stopped being
accordions — and the last report was the one none of that could reach, because
it was not a bug in the drawer. It was the drawer.

A drawer is a column. A column is in the layout. Below 900px Gradio wraps it to
a full-width block *above* the stage, which is a screenful of configuration on
top of the conversation and a composer somewhere under the fold — and the
mobile rules that made that survivable set the workspace to `height: auto`,
which gave up the one property the whole panel depends on. The old top bar was
the same mistake one level up: a mode selector, a model chooser, a rescan, Load,
Unload and a status line, above every workspace, at all times, wrapping to four
rows on a phone.

So Conversation is now what it always was: a messaging application. One
permanent screen, and everything else an overlay.

### 19.1 The screen

```
┌──────────────────────────────────────────────┐
│ ☰   Ada                              ● Loaded│  header, flex: 0 0 auto
│     harbour at night                         │
├──────────────────────────────────────────────┤
│                                              │
│  the transcript                              │  flex: 1 1 0, scrolls
│                                              │
├──────────────────────────────────────────────┤
│ Ready.                                       │  one line, never taller
│ 📎 [ Message…                    ] [  Send  ]│  flex: 0 0 auto
└──────────────────────────────────────────────┘
```

Three controls in the header: the menu, who you are talking to and about what,
and whether the model is up. Three in the composer: attach, the message, and
one primary action — **Send**, which becomes **Stop** in the same place while a
reply is streaming. Two components rather than one, because a single Gradio
button cannot carry two click handlers without both of them firing, and only
ever one of them is on screen or interactive.

The composer starts at one line and grows to six. The status line above it has
a minimum height, so a reply starting does not move the transcript by a pixel;
a long error scrolls inside it rather than growing.

### 19.2 Everything else is a sheet

`position: absolute` inside the workspace, which is `position: relative`. That
is the whole responsive contract, and it is one declaration rather than a media
query: a surface that opens takes no room in the layout, at any width, so
nothing it opens over can be pushed anywhere. The transcript stays mounted
behind it and comes back unchanged — same scroll position, same unsent message
— when it closes.

| surface | what is on it |
| --- | --- |
| Menu | Threads, Character, You; then Model / Runtime, Setup, Switch mode |
| Threads | search, the list, New, Rename, and Delete behind its own heading |
| Character | Talking to, Edit/New, the editor, card import, advanced sampling, Delete |
| You | your name, about you, Save |
| Message actions | the version pager and every per-message operation |
| Model and runtime | the chooser, Rescan, Load, Unload, the state, Open Setup |
| Workspace | the four modes |

Only one is ever open. That is not a rule somebody has to remember: it is
`_screens()` in `mc_llm_chat_panel.py` and `_sheet()` in `mc_llm_studio.py`,
each of which answers for *all* of its surfaces at once, and every handler that
opens one returns that whole answer rather than toggling a component of its own.

On a display wider than 900px the menu and the screens become a 23em left
sheet and the shell's sheets a corner panel, because a full-screen overlay for
a model chooser on a 27-inch monitor is a gesture rather than a layout. The
information architecture does not change with the width; only the geometry
does.

### 19.3 Editing borrows the composer

An edit used to open a six-line editor between the transcript and the composer,
which moved both of them. Now the action sheet closes, the composer is replaced
in place by an **Editing message** row with Save and Cancel, and the transcript
stays exactly where it was. Cancel restores the composer with whatever unsent
text was in it, because that text was never touched — it is a different
component.

The state of the edit row and of the composer are the last two entries in
`SELECTION_ORDER`, so *every* refresh of the transcript returns the panel to the
home state. There is no second handler that has to remember to.

### 19.4 The shell is a menu, a title and a state

The mode selector is still one `gr.Radio` over one list of modes — which mode is
open is a thing a radio says by construction — but it is in a sheet now, reached
from **☰** in the shell bar or from **Switch mode** in Conversation's own menu.
The model chooser, Rescan, Load and Unload are in the other sheet, reached from
the state chip in either header, from **Model / Runtime** in either menu, and
leading on to Setup for the residency, the estimator and the paths.

Prompt Studio, MiniMax H3 and Setup are untouched inside their own workspaces.
What changed for them is how you arrive: a menu instead of a row of pills, and a
state chip instead of a filename, a rescan and two buttons above everything they
draw.

While Conversation is open the shell bar is not drawn at all, because
Conversation's header carries the same three affordances beside the character
and the thread rather than above them. The two state chips are updated together,
from the runtime itself, after anything that can have changed it.

### 19.5 What is kept

Nothing in `prompt_master.chat` changed, and no handler's behaviour did:
`CharacterStore`, `ChatStore`, `_thread_choices`, `_load`, `_view`, the position
map, message versioning, branch, delete-from, continue, resend, the streaming
generator, cancellation, the character editor and card import, the persona, the
sampling fallbacks, the vision-projector refusal and the measured context size
are the same code. What was replaced was the presentation layer and the CSS
under it — wholesale, rather than by another override on top of the last one.

One behaviour is new rather than moved: pressing Enter in a one-line composer
now sends, because the message box is bound to `_send` through its `submit`
event as well as through the button. Ctrl/Cmd+Enter still submits and Escape
still stops, from `javascript/llm_studio.js`, which is also where the workspace
is still measured against the window into `--mc-llm-available`.

### 19.6 The sheet that would not close (19 August 2026)

The first build of the rebuild shipped with every sheet permanently on screen.
What that looked like: the model sheet floating over Setup with its ✕ doing
nothing, **☰** apparently dead, and no way back to Conversation — because the
mode sheet was there too, underneath the model sheet, and the model sheet was
covering the radio that would have got you out.

The cause is one declaration:

```css
#mc-llm-studio .mc-llm-sheet { display: flex; }   /* wrong */
```

Gradio hides a `Column` by putting a class on it whose rule is `display: none`
at *class* specificity. Any rule of ours that names `#mc-llm-studio` and sets
`display` outranks it, so `visible=False` had no effect on anything carrying
`.mc-llm-sheet`. The handlers were firing and their updates were being applied
the whole time — pressing **Open Setup** switched the workspace, which proves
it — and the element simply would not go away.

The mode views next door are the same component hidden the same way, have never
had a `display` rule, and have never failed to hide. That is the experiment run
both ways round in one screenshot.

The fix is to state no `display` at all: a Gradio Column is already a flex
column, so nothing was gained by restating it. `flex-direction` stays, because
it is inert unless something else makes the element flex.

This is also the real explanation for the report §15 answered with a button
label — "it opens but does not toggle close" was never a `gr.State` that had
drifted out of step. It was `#mc-llm-studio .mc-llm-drawer { display: flex; }`,
added one section earlier to stop the drawer expanding sideways, quietly
outranking the host's own way of hiding it.

Two things now hold the line:

- `tests/test_llm_panels.py` reads every container the panels build with a
  `visible=` argument, collects the `elem_classes` those containers carry, and
  fails if any rule under `#mc-llm-studio` sets `display` on one of them. The
  list is derived from the source, so a surface added tomorrow is covered the
  moment it exists.
- The menu buttons are toggles rather than openers, against a `gr.State` that
  names the open surface. A menu that can only open is one you cannot dismiss
  from the control you opened it with — and on a desktop, where the sheet is a
  corner panel that does not even cover the button, pressing it again looked
  like a dead control.

## 20. Krea 2 (20 August 2026)

A fifth workspace, and the first one added since the shell became a menu. It
writes Krea 2 image prompts with the local model: text alone, or text plus up
to four reference images.

Structurally it is MiniMax and not Conversation — one task in, one finished
prompt out, its own history, its own screen. It is explicitly *not* a
Conversation persona, not a variant of MiniMax and not an extension of LTX
Prompt Studio, because a prompt compiler and a chat are different products and
the tab has spent five sections learning not to merge them.

It generates no images. There is no sampler here, no CFG, no steps, no edit
LoRA strength, no grounding resolution, no style-reference strength, no
moodboard, no mask and no negative prompt. Those belong to an
image-generation integration; this is the thing that writes what such an
integration would be handed.

### 20.1 Whose words these are

The base system prompt is Krea's own `expansion.txt`, vendored byte-for-byte
from `krea-ai/krea-2` into `prompt_master/krea/` with its origin, its retrieval
date and its sha256 in `UPSTREAM_SOURCE.txt`. It is the file Krea's prompting
guide points at: *"If you wish to use LLM assistance for generating longer
prompts, check out expansion.txt and use it as a system prompt for LLM of your
choice."*

Nothing edits it. Everything this extension has to say about reference images
is a separate `REFERENCE_ADDENDUM` in `enhancer.py`, appended under a heading
that names it, so anybody reading an assembled system message can see where
Krea stops and we start. A text-only run gets upstream's file and nothing else,
which is also why it needs no vision projector and works on any model that can
hold a conversation.

The tests hold that line rather than trusting it: `test_llm_krea.py` asserts
the addendum is *not* in the upstream text, and that a text-only system message
is byte-identical to the vendored file.

One consequence worth stating, because it looks like a bug. Upstream's
instruction opens with *"Think step by step about the request before writing
the answer"* and then asks for the paragraph after the thinking. A model that
emits a visible `<think>` block is therefore obeying a file this package may
not edit, and the cleaner takes it off afterwards. The cleaner removes a
thinking block and a single enclosing code fence, and it is deliberately
incapable of anything else: a cleaner that improved prompts would be a second
prompt writer nobody can see, running after the one whose instructions are on
disk.

### 20.2 Image 1 has to stay Image 1

This is the whole feature, and everything else is arranged around it.

The request people actually make is *"replace the face of the woman in image 1
with the woman from image 2, keeping image 1's body, outfit, pose, framing,
lighting and background"*. That sentence means nothing at all if image 1 and
image 2 can trade places somewhere between the upload control and the finished
prompt — and it fails silently when they do, because what comes back is a
fluent paragraph describing the wrong edit.

So identity comes from the **slot**, and from nothing else. Not the filename,
not the upload time, not the temporary path, not the backend's tensor order,
not the caption text, and never from classifying what a picture appears to
contain. That rules out a multi-file upload control — a Gradio file list
reorders itself when an entry is deleted and replaced — so the references are
four explicit numbered slots.

A gap in the slots is **refused**, not closed up. Filling slot 1 and slot 3 and
pressing Generate gets a sentence saying Image 2 is empty; it does not quietly
promote the third picture, because the user has written "image 3" in their
instruction and would get a prompt about a picture the writer is calling
Image 2. The same rule governs a reference that cannot be described: the run
stops, naming the image that failed, rather than writing from the survivors.

The numbering is announced on the page, beside the slots, because the feature
rests on the user and the model agreeing which picture is which and an
agreement one side was never told about is not one. It is also announced as
*ours*: "Image 1" is an LLM Studio convention for talking to the prompt writer,
not Krea syntax, and the panel says so.

### 20.3 Caption first, in order, one at a time

The same shape MiniMax uses. Each reference is described on its own by the
vision model; the pass that writes the prompt is a text-only request over those
descriptions, laid out under their own numbers:

```
user_prompt:
Replace the face of the woman in image 1 with the woman from image 2.

reference_images:
Image 1: <description>
Image 2: <description>
```

The writer is never handed a row of unlabelled paragraphs and asked to work out
which is which — that is the failure this section exists to prevent, arriving
through the back door.

The captioning loop is sequential, and not for want of a thread pool: one
server answers one request at a time here anyway. It is sequential because a
sequential loop emits its captions in slot order *by construction*, which is
what lets the panel pair the first `CAPTION` event with Image 1 without either
side carrying an index around. §10 of the design intent allows exactly that,
and `test_llm_krea.py` tests the guarantee it rests on rather than assuming it.

The captioner is told to describe and nothing else: no naming real people, no
inferring what is out of frame, no deciding how the picture should be edited.
It is never shown the user's instruction. A captioner that knows what is about
to be asked for starts answering that instead, and then the writer writes a
prompt about the captioner's opinion.

### 20.4 A model that cannot see is told so

Text-only works anywhere. References require a vision projector, and when there
isn't one the run is refused **before** anything starts, with a sentence saying
what to do about it. It never falls back to text-only: silently dropping the
pictures would return a plausible paragraph about a request nobody made, and
the pictures were the request.

The refusal is stated twice on purpose — once in the panel, where nothing has
been started and nothing holds the GPU, and once where the client is actually
obtained, for anything that got past the first check.

Cancellation works during captioning and during writing, and the workload lock
comes back in a `finally` either way. There is no Krea-specific concurrency,
threading or GPU arbitration anywhere: it takes the same broker lock every
other run takes, for the same reason.

### 20.5 What history keeps, and what it refuses to keep

`krea-history.json`, its own file, alongside the others. A session holds the
request, the finished prompt, the seed, and two parallel ordered lists: the
reference **names** and the reference **captions**.

No image bytes. No data URLs. No temporary upload paths — a Gradio upload path
is a directory name nobody chose and a file name that means nothing, and a
history file that grew a base64 JPEG per entry is one nobody could open. A test
writes a session from a reference carrying pixels and a `/tmp/gradio/...` path
and asserts neither reaches the file.

Loading a session restores the text, the prompt and the captions, and says
which pictures it was written from — as information. It does not refill the
slots, because the files are not saved and may not exist. Writing another
reference-aware prompt means attaching them again, and the status line says so
rather than leaving it to be discovered.

### 20.6 The boundary this leaves open

Prompt synthesis and Krea image-edit conditioning are different systems, and
the seam between them is where reference order gets quietly redefined.

So a run does not reduce to one string. `prompt_master/krea/references.py`
keeps a `Reference` — its UI index, its path, its caption, and a
`semantic_role` that is only ever filled in from something the user said —
and a `KreaPromptResult` that carries the prompt *and* those references. A
future backend adapter may need them in a different order than the person
supplying them saw: a Forge Neo Krea identity edit may present subject-first
inputs while reordering scene and subject internally to match how the LoRA was
trained. That reordering is the adapter's to do.

What it may never do is redefine what "Image 1" meant to the user, or to the
prompt it is handed alongside. User order is the semantic source of truth;
backend order is a detail of whatever eventually draws the picture. Version 1
ships the first half and leaves the second unclaimed.

## 21. Two ways to lose something (20 August 2026)

Reported together, from one afternoon and one `llama-server.log`: the card was
full while the tab said *Unloaded*, and the first replies of every old thread
came back blank.

They are unrelated, and both are the same shape of bug — a thing that exists
in memory and never reaches the place that outlives memory.

### 21.1 What the log actually said

41 starts. 19 loaded a model, 22 died on `cudaMalloc failed: out of memory`.
Two things in that are worth reading twice.

The first is a staircase. Four consecutive starts, all successful, all reading
free VRAM on the way in:

```
free = 3078 MiB   →   2963   →   2868   →   2766
```

Roughly a hundred megabytes gone per cycle and never coming back. That is about
the size of a CUDA context, and those four starts are the *move the model to
system RAM* path being exercised four times — which is where the leak below
lives.

The second is nineteen starts in a row failing to allocate while the device
reported **23304 MiB free**, the same figure to the megabyte on 33 of the 41
starts. A number that never moves is not a measurement. The genuine readings in
the same file (22987, 7493, 4744, 3745, 3078, 2963, 2868, 2766) all differ from
each other, as real ones do. So on this machine llama.cpp's own free-VRAM line
is not a live reading, and an 18 GB allocation failing under it tells you only
that *something else has the card* — not what.

The retry ladder is visible too: 18231 MiB → 15244 → 10588, three attempts,
then the start gives up. All three failed, every time.

### 21.2 The server that outlived its handle

`Runtime._restart_in_system_ram` moves the model off the card without losing
the loaded server: stop the old process, start a new one with no GPU layers,
wait for it to answer. Its failure branch logged *"could not move the LLM to
system RAM; stopping it instead"* and stopped nothing.

`start()` can succeed and `wait_ready()` still fail — a server that came up and
then died, or one slower than the timeout. What was left behind was a live
`llama-server` that nothing had a handle to any more. It is launched into its
own process group and with `CREATE_NO_WINDOW`, which is exactly right while the
WebUI owns it and exactly wrong once it does not: it outlives its parent, holds
its CUDA context and its weights, and is invisible anywhere except Task
Manager.

From the tab, that is a card with no free VRAM, a chip reading *Unloaded* —
truthfully, about the runtime it knows about — and an Unload button that
correctly reports it has nothing to stop. Forge's own unload cannot reach it
either; it is not Forge's process.

The main start path has always stopped its process on this failure. This one
now does too.

### 21.3 Unload means the card, not the handle

Fixing the leak stops new strays. It does nothing about the ones already
running, and *"my VRAM is full"* is not a problem you solve by shipping a fix
that only helps next time.

So `mc_llm_runtime.strays()` looks for `llama-server` processes whose `--alias`
is the one this extension passes, minus the one this WebUI owns, and Unload
stops them and says what it found:

> Also stopped 1 stray llama-server left running by an earlier session,
> releasing 15.9 GB.

From a press rather than at startup, deliberately. Two WebUIs sharing one card
would each see the other's server as a stray, and a startup that quietly killed
it would be a worse bug than the one being fixed. A press is somebody asking
for their card back.

The alias lives in a vendored file that is not ours to edit, so the constant is
restated in `mc_llm_runtime` and a test asserts the two still agree. Matching is
on the *value* of `--alias`, not on the word appearing anywhere on the command
line: a model kept in a folder of that name would otherwise make every server
on the machine look like ours.

### 21.4 The reply that was on screen and not on disk

`_stream` writes a reply into the conversation and saves it, on four paths:
finished, cancelled, failed, and raised. There is a fifth, and it was the one
Stop used.

Stop is wired as `cancels=`, and what Gradio does with that is *close* the
handler's generator. Closing raises `GeneratorExit`, which derives from
`BaseException` and goes straight past `except Exception` — so the branch
holding the only `store.save` never ran. The reply stayed on screen, because
Gradio keeps the rows it was last given; it was never written to the thread. It
came back missing the next time the thread was opened, and `_view` rendered
what was left as a message of yours with nothing under it. A browser refresh
mid-reply and a dropped queue entry went the same way.

The save now happens in a `finally`, through one idempotent `keep()` that every
exit goes through. Saving is safe there where yielding would not be: yielding
during a `GeneratorExit` is a `RuntimeError`, writing a file is not. What gets
saved is whatever the message holds — a partial reply is a real reply, which is
what the CANCELLED branch had already decided, and `_tidy` still clears up a
reply that produced nothing rather than leaving an empty bubble in the thread.

This is the same lesson `mc_llm_sessions._traced` learned about `GeneratorExit`
and the GPU lock, arriving a second time about a different resource. The rule
generalises: **anything a streaming handler owns and must not lose belongs in a
`finally`, because the way these generators most often end is the way that
skips every `except`.**

The test for it closes the generator after one chunk and reads the thread back
off disk. Neutering the `finally` fails that test and only that test.

## 22. Managed backbones (20 August 2026)

LLM Studio has always run any GGUF on the machine. That is the right floor and
it is a poor first experience: a new installation has no weights, no way to know
which of several thousand models this application was written against, and no
way to find out except by downloading a few and comparing. The catalogue is the
other half — six backbones chosen in advance, each with the settings it should
run at already decided.

The design intent for it is `llm_studio_managed_backbones_design_intent.txt`.
What follows is where the implementation matches it, and the three places it
does not.

### 22.1 The registry is the trust root, and it is the only reachable thing

`prompt_master/models/managed-models.json` is checked in, reviewed like source,
and carries a SHA-256 for every byte it names. `mc_llm_managed_models.entry()`
is the only route to a download and it accepts an id from that file or nothing,
so there is no code path from the UI to an arbitrary URL — not because the UI
declines to offer one, but because no function underneath it takes one.

Every row is validated at load: the id against a regex with no separators in it
(the id becomes a directory name, so that regex *is* the traversal defence), the
filename as a plain `.gguf`, the source as HTTPS, the hash as 64 hex characters.
These are checks against a file inside the extension, which is exactly why they
are cheap enough to be worth having.

The publisher's filename never becomes a path. A bundle on disk is always
`model.gguf`, `mmproj.gguf` and `installed.json`; the real name is *recorded* in
the manifest so a bundle can say what it is, and is used only to build the URL.

### 22.2 Shipped on a branch, and why that is not the hole it looks like

The design intent says to pin an immutable revision. The six entries ship with
`"revision": "main"` and `"bytes": null`, because the machine this was written
on cannot reach huggingface.co and neither commit shas nor exact byte counts can
be invented.

That is a real gap and it is not a security one. The trust anchor is the
SHA-256 committed here, and it is checked after every transfer, so a publisher
who re-uploads gets a **refusal**, never a substitution — which is the property
§7.3 actually asks for. What a branch costs is the *quality* of that refusal: a
moved `main` fails as a hash mismatch and a sentence asking for an extension
update, where a pinned commit would simply have kept working. The panel says
which of the two an entry is.

`tools/pin_managed_models.py` closes it in one run on a machine with network
access. It resolves the commit and the LFS object sizes, and it **never writes
the hub's hash over the one checked in** — a disagreement fails the whole run
with a sentence, because a publisher whose files have really changed is a review
decision and not something a script does at three in the morning. `bytes` stays
optional either way, which is the precedent `release-manifest.json` already set:
the size is used for the disk-space check and the progress fraction, and the
hash is what decides whether a file is the right file.

### 22.3 The transaction, and the reason it reuses the vendored downloader

Stage under `managed/.downloads/<id>/`, verify, write the manifest, and rename
the directory into place. Until that rename the installed tree does not know the
download exists, which is what makes every interesting failure boring: a
cancelled download is a `.part` file, a corrupted one is a deleted `.part`, a
crash is a staging directory nothing reads, and in all three the model selected
a minute ago is still selected, still on disk and still startable.

The transfer itself is `prompt_master.provisioning.downloader`, unchanged. It
already has HTTP Range resumption, a retry budget that only counts attempts
which moved no bytes, the 416-means-start-again rule, and a verify-then-rename
that keeps a failed hash from ever becoming a file with a real name. Writing a
second downloader to gain a catalogue would have been two of them to keep
correct. What this module adds around it is the sidecar: a `.part` file is a
pile of bytes with no memory of what it was going to be, so the expectations are
written down first and a staging directory whose sidecar does not match the
entry being downloaded *now* is discarded rather than continued. Appending the
current quantisation to the previous one's prefix produces a file of exactly the
right length that is not any model at all.

Two checks the hash cannot make are made anyway. Free space is checked against
the *remaining* bytes plus `max(512 MB, 5%)`, before anything is created, so a
refusal is a sentence at the start rather than an `OSError` at 94%. And every
verified artifact has its GGUF header read: a file can match its SHA-256
perfectly and still be the wrong kind of thing, which without this would surface
as a llama-server startup failure several minutes and one discarded model later.

Promotion moves a previous bundle aside and puts it back if the rename fails —
the pattern §3 of `mc_llm_setup` established for the runtime, for the same
reason. `rmtree` walks a directory file by file, so deleting first would leave
an installation that had a working model with half of one.

### 22.4 Applying is a different transaction with a stricter rule

Downloading eight gigabytes while a model is loaded is fine and deliberately
does not disturb it. Swapping which model is *resident* is the operation with
the hard invariant: at no point may two llama-server processes intentionally
hold weights.

So a switch refuses first (an LLM generation in flight, or the GPU held by an
image job, are both a sentence rather than a queue), snapshots the state, stops
the server and then **waits until it is observed gone** — `stop()` returning is
a statement about a handle, and on Windows the server can still be unmapping a
17 GB file for several seconds — writes the new selection atomically, starts the
new backbone through the existing placement negotiation, and runs one eight-token
completion. A server that reaches `/health` has loaded a file; it has not
necessarily loaded a chat model or applied a template, and without the smoke
test the first thing to discover that would be somebody's real generation, with
the previous model long since discarded. Any failure after the snapshot restores
it and restarts what was there, keeping the downloaded files.

### 22.5 A profile is not a settings object

`prompt_master/models/managed_profiles.py` holds them as constants. `config()`
reads three layers as it always did — state file, preferences, Settings page —
and then, for a managed backbone only, a fourth that wins. That precedence *is*
the feature: somebody who chose "Gemma 4 12B QAT Balanced" chose an 8192
context, a q8_0 cache and a chat-template flag along with it, whether or not
they know that, and leaving the Settings page authoritative would run a curated
model at whatever the previous one happened to need.

It reaches exactly as far as the profile's own fields. No profile names a GPU
index, forces an offload or asks for mixed mode, and the dataclass has no field
in which one could: **profiles control model behaviour, the broker controls
where the model fits.** `context_mode` is forced to `fixed` rather than left on
`auto`, because automatic sizing spends whatever VRAM is free on context and
8192 is a decision about this workload rather than a floor to grow from —
negotiation may still shrink it to fit, which is a report and not a setting.

Temperature and top_p are deliberately absent from `SAMPLER_FIELDS`. They arrive
per request from Creative Mode's 0–10 curve, and a checked-in file that set them
would be overriding the user's own Creativity slider. The remaining fields go
out in the request body, filtered against the same whitelist at both ends —
"whatever the profile dict contained" is not a thing to put in a JSON payload.

The profile id is part of `_identity()`, so switching backbones restarts the
server even when the two agree about context and cache: `--jinja` and the cache
types are start-time arguments, and a running server cannot be told about
either. Without it, a switch between two similar models would have reused the
process that was still holding the previous weights.

### 22.6 What a manual install gets, which is nothing

`_profile_arguments` returns `{}` without a profile and the llama-server command
line is byte-for-byte the one this extension has always built. That is a
decision rather than an omission, and it leaves one thing visibly odd: the
Settings page's KV cache types have never been passed to llama-server at all —
they feed the VRAM estimate and nothing else. Quietly starting to honour them
here would have changed both the quality and the footprint of every existing
install as a side effect of adding a catalogue. It is worth fixing on its own
terms; it was not worth fixing inside this one.

### 22.7 The two routes had to know about each other

A managed bundle lives under the LLM data root, which is also where the ordinary
model chooser scans, so a downloaded backbone appears in that list like any other
GGUF the moment it arrives. Picking it there used to be indistinguishable from
picking a stranger's file — which would have silently dropped the hidden profile
that is the entire reason to have downloaded it.

`follow_path()` is called by both manual routes and settles it in both
directions: a path inside a bundle restores the managed selection, anything else
clears it. Without the second half, a profile written for one model would be
applied to another — an 8192 context and a q8_0 cache imposed on a file nobody
measured for them.

### 22.8 What is on screen

One dropdown, one line, one button, and a link to the model card. The button
reads **Download & Use** or **Use**, so what pressing it costs is on the button
rather than in a dialog afterwards. The line is role, size, family and state
(*Not downloaded*, *Download interrupted*, *Installed*, *Installed — older
revision*, *Active*), and nothing else: no temperature, no top-k, no cache type,
no template flag, for managed models. The test for that reads the labels off the
panel Setup actually builds, so a control added later fails there rather than
being noticed by a user.
