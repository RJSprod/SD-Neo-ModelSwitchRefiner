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

The deciding is one-directional, and this is the direction §18 forces. A
foreground image pass can reclaim an idle llama-server's VRAM, because
"ordinary txt2img/img2img remains functional" outranks any placement the LLM
would have preferred.

The reverse never happens. §13 used to make it a setting — three policies, one
of which ("LLM priority") demoted image residency to pay for a placement — and
the setting is gone, because in practice there is no case for it. The image
model is the workload the user is waiting on; the language model writes a
prompt for it and then has nothing to do. A checkpoint evicted for a prompt is
wanted again seconds later, so the borrowed VRAM is paid for twice, and on a
24 GB card that arrived in a user's console as `Moving model(s) has taken 5.92
seconds` followed by `Moving model(s) has taken 8.07 seconds` — thirteen
seconds added to every press of Generate to undo an eviction that had bought
the writer a few seconds. `_victim_order` now answers `()` for the LLM in both
residency modes, and `negotiate` shrinks instead of asking.

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

### 3.5 Reduction instead of displacement

§13 asks for "whichever residency creates the least disruption" to be demoted.
With displacement of the image side off the table (§3.3), what is left is an
order in which the LLM gives up its *own* resources, and it is the order of
increasing cost: the context first, because a cache nobody has filled is the
cheapest thing on the card to give up and a 128k context sized automatically to
fill free VRAM is frequently exactly that; then, for a mixture-of-experts
model, the experts, which are most of the weights and are consulted two at a
time; then blocks, four at a time; then the whole model in system RAM.

Every rung is reported, which is §13's other requirement and the reason the
ladder is legible from a console log rather than only from a debugger.

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
`test_llm_files.py`, `test_llm_accel.py` and `test_cross_workload.py`. They run
without a GPU, a model file or a WebUI.

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
  and a synthetic six-block header exercises it in a millisecond;
- `test_llm_accel.py` asserts, more than anything else, a **reclaimer that was
  never called**. Cooperative memory is not "little was released", it is
  "nothing was asked for", and a test that checked free VRAM afterwards would
  pass against an implementation that asked and was refused.

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
header, and asks the image side for nothing at all. That last part was a real
change in Exclusive mode, which used to sweep the image family before every
message: the sweep was narrowed to the moment the LLM is *placed*, which is
when ownership of the card would actually have changed hands. Re-sweeping
before a message about to be answered out of VRAM the LLM is already holding
evicts a checkpoint to buy nothing.

§25 finished the job: the LLM does not sweep the image family when it is placed
either, or ask for its VRAM in any other way. What is described below as "the
image side has its own way of asking" is now the only direction that asks.

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


## 23. CPU placement could not start at all (21 August 2026)

Reported as *"Creative Mode is on and my prompt is not changing"*, with a
`llama-server.log`. The log answered it in three lines, repeated twenty-three
times:

```
E llama_prepare_model_devices: invalid value for main_gpu: 0 (available devices: 0)
E llama_model_load_from_file_impl: failed to load model
E srv  llama_server: exiting due to model loading error
```

Every one of those starts enumerated one device — the CPU. The last start that
worked had listed `CUDA0` and a 3090, loaded, answered one request and stopped;
everything after it was this. In between, the installation's placement had been
changed to **CPU (system RAM)** in Setup.

### 23.1 Two halves of one command line disagreeing

`prompt_master/inference/llama_process.py` writes the same flags every time:

```
--device none --split-mode none --main-gpu 0
```

`--device none` is llama.cpp's own token for *no devices* — CPU placement also
empties `CUDA_VISIBLE_DEVICES`, so the process genuinely has none.
`--split-mode none --main-gpu 0` is *use device number 0*. Both have been in
that line since the tree was vendored, and llama.cpp used to ignore the second
when the first had emptied the device list. It validates it now:

```cpp
if (params.split_mode == LLAMA_SPLIT_MODE_NONE) {
    if (params.main_gpu >= (int)model->devices.size()) { ...error... }
```

So the failure is not a degraded language model, it is no language model, and
every surface that needs one is affected at once. LLM Studio says llama-server
exited before becoming ready. Creative Mode does what it is designed to do when
the writer will not answer — generate the prompt as typed — which is why the
symptom that got reported was "my prompt is not changing" rather than "the LLM
is down".

### 23.2 The fix is not in the vendored file

`prompt_master/VENDORED_FROM.txt` is explicit: *"Do not hand-edit these files —
changes belong in the mc_llm_* modules that sit on top of them."*

The command is assembled inside `LlamaProcess.start`, so a subclass overriding
that method means copying the whole assembly into this repository, where it
would silently stop matching the next version of the vendored tree. Instead
`mc_llm_runtime._Launcher` stands in for the `subprocess` name inside that one
vendored module: it forwards every attribute untouched and rewrites exactly one
command shape on its way to the operating system, via
`without_gpu_selection()`. The vendored file stays byte-identical and `diff -r`
stays clean.

`without_gpu_selection` only ever removes `--split-mode` and `--main-gpu`, only
when `--device none` is in the same line, and says so in the log when it does.
A GPU or Mixed placement is untouched, and a command that is not llama-server's
— a device probe spawned while a server is starting — passes straight through.

### 23.3 Two things that made it hard to see

**The log said the symptom, not the cause.** `read_failure()` reports
llama.cpp's own last words, and its `failed to load model '...'` pattern matched
first — true, and useless. It now recognises the device-selection failure
specifically and says what produces it: a CPU placement on an extension that
predates this fix, or a card that is not reaching llama.cpp at all
(`CUDA_VISIBLE_DEVICES` set in the environment, a runtime build with no CUDA
backend beside it).

**Creative Mode fell back in silence.** Falling back to the typed prompt is
right — a writer that will not answer is not a reason to refuse a generation
somebody asked for — but it was said only in the console, and what a user sees
is an image made from their four words with nothing indicating that anything was
meant to happen. `ScriptKreaCreative.postprocess` now puts the reason on the
result, beside the image, where this extension already puts the sentence when
Stage 2 fails.


## 24. Mixed placement was a mode that did nothing (21 August 2026)

Asked for directly: *"can you make it so mixed mode keeps some parts on GPU such
that only what is necessary to make the GPU run the LLM is loaded"* and *"can we
apply sage or other speed ups for GPU only and mixed mode?"*

### 24.1 What Mixed was

`Config.__post_init__` forced `gpu_layers = NO_OFFLOAD` for `MIXED_MODE`,
`Config.on_gpu` was therefore False, and `negotiate()` returned at its first
branch — *"System RAM or CPU execution: there is no VRAM decision to make"*. The
command line got `--n-gpu-layers 0`, llama.cpp put nothing on the card, and the
mode's own description said "card used for processing".

The whole partial-offload ladder already existed and Mixed was the one mode
forbidden from reaching it.

### 24.2 What Mixed is

`_requested_placement` now asks for every layer with `on_gpu=True` when the mode
is Mixed and a card is configured, and the ladder does the rest: shrink the
context, move the experts, drop blocks, and land on zero — today's behaviour,
unchanged — when nothing is free. Mixed originally forced a preserve-image
policy for its own call, to be sure the middle option could never cost somebody
their checkpoint; §25 made that the rule for every placement, so the special
case is gone and the guarantee is wider than it was.

Two properties are worth stating because they are what make this safe to do by
default:

- **It never asks anything to move.** Preserve-image never calls the broker, so
  no checkpoint is evicted and no image generation is slowed to make a prompt
  faster.
- **It sizes against the image model's needs.** A Creative roll already passes
  `image_reserve_bytes()` as `extra_reserve`, so what Mixed fills is what is
  spare *after* the picture that is about to be generated has been accounted
  for.

### 24.3 Moving the experts instead of the blocks

`_shrink_offload` stepped down in fours: 40 layers, 36, 32… Each step moves a
whole block, which is that block's attention *and* its feed-forward. For a
mixture-of-experts model that is the wrong thing to give up first: the experts
are the great majority of the weights and are consulted a couple at a time,
while attention is small and every token touches it.

So for an MoE model the first degradation step is now `--cpu-moe` — all layers
resident, experts in system RAM — and only if that still does not fit do blocks
start leaving. Simulated against a 26B-A4B's real shape:

```
6.4 GB free →  all 40 blocks on the GPU, experts in system RAM   (3.7 GB)
3.0 GB free →  28 of 40 blocks, experts in system RAM            (3.0 GB)
0.5 GB free →  system RAM                                        (as before)
```

`Placement.cpu_experts` carries it, `weights_bytes` prices it from
`Gguf.expert_share` (estimated from the header's expert count and widths, since
the tensor list is past the metadata block `mc_gguf` promises not to read), and
`Placement.key` includes it so calibration does not file two different
footprints under one identity.

### 24.4 Asking the build rather than assuming

`--cpu-moe` and `--flash-attn` were both added to llama.cpp at different times,
and `--flash-attn` changed from a switch to `on|off|auto` along the way. The
runtime is whatever build the user copied in, and a flag it has never heard of
is not a slower server but a server that exits at startup — the failure this
extension already spent a week on (§23).

So `runtime_capabilities()` runs `llama-server --help` once per binary, caches
the answer against the executable's modification time, and reduces it to the set
of long options it advertises. `accelerator_flags()` adds only what is in that
set, only for a placement that puts work on the card, and passes `--flash-attn`
its value only when the help text shows the three-state spelling.

They reach the command through `_Launcher`, the same stand-in that removes the
impossible flag pair, because the vendored `LlamaProcess.start` has a fixed
keyword list with no room for arbitrary arguments and
`prompt_master/VENDORED_FROM.txt` forbids editing it.

### 24.5 What was not added, and why

Sage attention is a quantised attention kernel for diffusion models in PyTorch.
There is no llama.cpp counterpart and nothing to enable. The remaining llama.cpp
knobs are thread counts and batch sizes, which are hardware guesses — on a
hybrid Intel part the right thread count is often *fewer* than the default, and
which is right cannot be determined from here. Now that every request's measured
tokens per second is kept per backbone (§23.3), that is the shape a future
answer should take: measure, then choose — not guess in a docstring.


## 25. The language model was evicting the image model (21 August 2026)

Asked for directly, and twice in one message: *"i see the image model is being
moved in and out of vram. it should always stay!"* and *"The goal is to always
prioritize image model in vram over LLM."*

### 25.1 What the log said

A 24 GB card, a 13.9 GB checkpoint, a 12B model in Q4_K_M, Exclusive mode. Every
press of Generate produced this:

```
released 13.9 GB of image VRAM for the LLM taking VRAM ownership in Exclusive
mode (8.7 GB -> 22.6 GB free)
demoted the image checkpoint (kromaV02TurboINT8_v02) (13.9 GB)
starting llama-server — 16 layers on the GPU, 22.5 GB free
...
llama-server stopped — the image generation that follows a Krea roll
Moving model(s) has taken 5.92 seconds
Moving model(s) has taken 8.07 seconds
```

Three separate faults, each of which made the next one worse.

### 25.2 Fault one: Exclusive mode swept in both directions

`request_vram` read the mode and swept "the other family", whichever family was
asking. For an image pass that is the mode's promise. For an LLM it was a
13.9 GB eviction to make room for a model that had asked for about ten — and the
checkpoint was needed again a second and a half later, so the generation paid
`5.92s + 8.07s` to move the same weights back. Every press.

The sweep is now the image family's alone: `sweeping = exclusive_sweep and
family == FAMILY_IMAGE and mode() == MODE_EXCLUSIVE`. Backing it up,
`_victim_order` answers `()` for the LLM under both modes, so nothing anywhere
can demote an image residency for a language model.

### 25.3 Fault two: the reserve did not know the checkpoint was there

`image_reserve_bytes` is how a Krea roll tells `negotiate` to leave room for the
generation that follows. It subtracts what the image family already holds,
because reserving that a second time would shrink the writer to make room for a
model already on the card — and it read that figure from
`mc_broker.resident_bytes`, the residency *register*.

Image checkpoints are never in the register. They are loaded and moved by Forge,
and `mc_memory` cooperates with that rather than announcing every load to the
broker; the honest answer comes from asking the reclaimer, which is what
`reported_bytes` does and what `status()` had always used. So the register said
0 for a checkpoint sitting in fourteen gigabytes of VRAM, the roll reserved
13.9 GB for it all over again, and the writer was sized against 8.7 GB *minus*
13.9 GB. That is why a 3090 with room to spare ran sixteen of forty-eight blocks
on the card.

`mc_broker.held_bytes(family)` is now the one function to ask when the question
is "is it already there", and it is `max(declared, reported)`.

### 25.4 Fault three: the two faults fed each other

`hand_back_vram` runs after a roll and asks for `required - held`. With the
checkpoint swept off the card by fault one and `held` misread by fault two, it
asked for the full 13.9 GB — which in Exclusive mode swept llama-server out, so
the next roll started a cold server. Time to the first character in that log was
28 seconds. With the checkpoint resident and counted, the call is a no-op, the
server stays up, and the next roll is warm.

### 25.5 What the policy setting became

`OPT_POLICY` had three values and all three answered the same question: what may
the LLM take from the image side. The answer is now "nothing", so the setting
was removed rather than left on the Settings page describing a choice it no
longer makes. `Status.policy` is gone with it, and LLM Studio's residency panel
states the rule instead of reporting a setting.

What remains configurable is the direction that still has two honest answers —
what happens to llama-server when a generation starts:

| Mode | Behaviour |
| --- | --- |
| **Hybrid** (default) | It is left alone. It is holding spare VRAM, the generation is unaffected, and the next prompt starts warm. |
| **Exclusive** | It is stopped and the generation gets the whole card. More headroom for one large pass, at the cost of a model load per image. |

An Exclusive sweep that actually stops a server now says which setting would not
have, once, in the console — because "a model load per image" is a real cost and
the mode was chosen, in this user's case, to fix the very problem §25.2 was
causing.

### 25.6 Rewording a radio label must not reset it

`MODES`' labels changed in this work, and what a Gradio radio stores is the
whole displayed string. `resolve` therefore falls back to matching the *naming*
half of a label — everything before the em dash — so an installation holding
`Exclusive — one family owns VRAM at a time` still resolves to `exclusive`
rather than silently reverting to the default.

### 25.7 `preferences.json` and `[WinError 5]`

Visible in the same log, unrelated to VRAM: `os.replace` is atomic on POSIX and
merely usually-atomic on Windows, where a scanner or indexer holding the
destination open for a moment answers `Access is denied`. The write gave up on
the first refusal, so a preference was lost and a traceback reached the console.
It now retries five times over a widening backoff, and removes the temporary
file when it truly cannot land — the log also showed a leftover
`preferences.jsonfyz9tar1.tmp` in a directory users open to read their settings.


## 26. Twenty seconds before the first character (21 August 2026)

Six runs from one session — three on Mixed, three on the processor — and two
observations, both correct:

> "Starting llama-server…" appears often. After a first run on a mode, it should
> never have to start again.

> Look at those numbers for CPU mode, it takes over 20 seconds to get to the
> first character from LLM even when warm.

### 26.1 Why it kept restarting

§25 stopped the LLM from evicting the image model. It did not stop the image
model from evicting the LLM, and with VRAM residency set to Exclusive that
happened on every roll:

```
llama-server stopped — the image generation that follows a Krea roll, 4.4 GB released
the image generation that follows a Krea roll: demoted the LLM (Q4_K_M) (4.4 GB)
```

The request behind that was `hand_back_vram`, asking for `required − held` —
about 2 GB, with 4.3 GB free. It fitted. Exclusive mode swept anyway, because
that is what Exclusive mode meant: ownership rather than arithmetic.

That meaning is now empty. The LLM is confined to spare VRAM by construction
(§25) and a Krea roll reserves the coming pass its full requirement before it
places itself, so "the image family owns the card" describes a state that is
already true. What the sweep still did was cost a model load per image — and,
worse than the load, llama.cpp's prompt cache, which is where the next section
begins.

So the setting was renamed to the only thing it still decides:

| Setting | Behaviour |
| --- | --- |
| **Keep the LLM loaded** (default) | llama-server stays in the spare VRAM. The generation is unaffected and the next prompt starts warm. |
| **Free the LLM for every image** | llama-server is stopped and the pass gets every last byte, at the cost of a model load per image. |

The constants behind them are still `hybrid` and `exclusive`, so a config
holding either value resolves. A config holding an old *label* does not, and
falls back to the default deliberately: "one family owns VRAM at a time" was an
answer to a question that no longer exists, and carrying it onto "stop
llama-server for every image" would be carrying a preference about one thing
onto another.

A pass that genuinely does need the room still gets it. `mc_memory` calls
`_reclaim_for_image` when its own eviction falls short, and that path demotes
the LLM on evidence rather than on principle.

### 26.2 Why a warm roll still cost twenty seconds

From the llama-server log, one processor-only server, three consecutive rolls:

```
task 5   | prompt eval time = 35337 ms / 1065 tokens
task 199 | prompt eval time = 21335 ms /  601 tokens
task 405 | prompt eval time = 20258 ms /  596 tokens
```

Nothing was wrong with generation — 4.8 tokens a second, as expected from system
RAM. The twenty seconds were *prefill*, and the reason it is 600 tokens rather
than 60 is two separate things.

**The first is that a checkpoint, not the prefix, decides where a warm turn
resumes.** llama.cpp's own accounting says so:

```
slot get_availabl: selected slot by LCP similarity, sim_best = 0.630
slot update_slots: Checking checkpoint with [0, 1061] against 0...
slot update_slots: Checking checkpoint with [0, 460] against 0...
slot update_slots: restored context checkpoint (pos_max = 460, n_past = 460)
slot update_slots: erased invalidated context checkpoint (n_swa = 1024)
```

668 tokens matched (`sim_best = 0.630` of 1,060). 460 were resumed. The other
208 were read again, because Gemma is a sliding-window model, most of its blocks
keep only a 1,024-token window of the cache, and a prompt can therefore only be
resumed at one of the checkpoints taken on the way past. `--swa-full` keeps the
whole cache, which removes both the window and the checkpoints: llama.cpp then
resumes at the true common prefix. It is added for every placement, processor
included, because what it buys is reuse rather than throughput — and the memory
it costs is memory `mc_llm_context.estimate` has always charged for, since that
function sizes the cache at the full context for every block.

**The second is that the first roll on a server pays for the instruction.** 1,065
tokens against 601: the difference is Krea's standing instruction, which is the
same text every time and is known before anybody presses anything. There is no
reason for a person watching a progress bar to be the one who reads it, so it is
prefilled when llama-server starts, on a thread nobody is waiting on, with a
one-token completion whose answer is discarded.

What makes that safe to fire from *any* start — including a re-placement in the
middle of a roll — is the workload lock. The prime asks for the GPU as
background work and does not wait, so a start that some job is holding the card
for finds it taken and does nothing at all. The roll it would have queued in
front of is precisely the roll it existed to help.

### 26.3 What is left, and what it is not

On the machine those logs came from, a warm processor-only roll should now
prefill about 390 tokens rather than 597, and a warm Mixed roll about 390 rather
than 1,050 with no server start in front of it — eighteen seconds to the first
character becoming roughly three.

The 390 that remain are the creative brief, and they are genuinely new text: a
different roll of the axes is a different brief. Nothing in the plumbing can
cache what has not been written yet. Below that number the levers are the
placement (126 tokens a second on Mixed against 30 on the processor, so the same
brief is three seconds or thirteen) and the size of the brief itself, which is
what the Creativity slider and the number of non-Natural axes control.

## 27. Three more weights of one backbone, and a ladder with rungs (21 August 2026)

Design intent: `gemma26b_quant_tiers_speed_design_intent.txt`.

The managed catalogue shipped one 26B-A4B entry, at Q4_K_M, which is 16.8 GB of
weights plus a 1.19 GB projector. That is a backbone for a 24 GB card with
nothing else on it, and the machines this extension runs on have a Krea 2 stack
on the same card. The answer is not a different model — the whole reason the 26B
is in the catalogue is that it writes prompts the way this application wants and
generates fast for its size, because a mixture-of-experts activates about 4B of
its weights per token. The answer is the *same* model at fewer bits.

### 27.1 Three entries, one backbone

`prompt_master/models/managed-models.json` gains Q4_K_P (~16.9 GB), Q3_K_P
(~13.4 GB) and Q2_K_P (~10.7 GB) from the publisher this catalogue already
names, pinned to commit `96c11c22`. Q3_K_P is the one marked as the recommended
26B; Q2_K_P is labelled *Low memory* and nothing in the catalogue calls it
equivalent, because it is not.

The existing Q4_K_M entry keeps its id, its hashes and its revision. That is not
tidiness: the id is what an installed state file names and the hashes are what
`Installed.matches` compares, so anything else would have shown every existing
26B install as *Installed — older revision* and invited a re-download of files
that had not changed. The three new entries are pinned; the shipped one is left
exactly as the installations that have it expect to find it, and
`tools/pin_managed_models.py` remains the way to close that on a machine with
network access, which is also where the `bytes` fields get filled in.

Each tier gets its own hidden profile in `managed_profiles.py`. The only thing
that differs between them is the KV cache: f16 for the quality tier, because
that tier exists to be as close to the known Q4 baseline as a checked-in file
can make it, and q8_0 for the other two, because the cache is the term that
scales with context rather than with the file and is therefore the cheapest
gigabyte on the card to buy back. No tier sets a sampler. A different number of
bits per weight is not a reason to move a temperature, and stacking two lossy
decisions on the low-memory tier would leave nobody able to say which of them a
bad caption came from.

**One projector, four names.** All four tiers name the same
`mmproj-…-f16.gguf`, byte for byte and hash for hash. Installing all four used
to mean fetching 1.19 GB four times and keeping four copies of it. The
downloader now looks for an installed bundle whose manifest records exactly the
hash it wants, re-reads that file's digest — the manifest describes the past and
this is somebody's disk — and hard-links it into the new bundle, copying where
the filesystem will not link. Every failure in that path returns "no" and the
file is downloaded exactly as it always was, because a disk optimisation that
can fail an install is not one worth having.

### 27.2 `--cpu-moe` was a cliff

The residency ladder already preferred moving an MoE model's experts over
dropping whole blocks, and for the right reason: experts are the great majority
of the weights and are consulted a couple at a time, while attention is small
and every token touches it. What it had was one flag to say it with.
`--cpu-moe` moves *every* expert. So a model two gigabytes too large for a card
answered that by reading thirty-four blocks' worth of experts across the bus for
the rest of the session, when six blocks' worth would have closed the gap.

llama.cpp has `--n-cpu-moe N`, which keeps the experts of the first N blocks in
system RAM and leaves the rest where they are. `Placement.cpu_experts` was a
boolean and is now `cpu_expert_layers`, an integer with three meanings: nothing
moved, N blocks moved, or the `ALL_EXPERTS` sentinel that still means
`--cpu-moe`. `cpu_experts` survives as a property, because "are any of them
elsewhere" is the only question most callers were asking.

The count is **solved rather than searched**. The saving is linear in the number
of blocks moved — `expert_share × N / block_count` of the file — so
`_minimum_expert_layers` measures what moving all of them would save, divides the
shortfall by the per-block share of it, and returns one number. The caller
verifies that number against the real arithmetic anyway, and steps it up if the
estimate was optimistic. The ladder in `_shrink_offload` is then:

1. `--n-cpu-moe N`, for the smallest N that fits;
2. `--cpu-moe`, once N has reached every block anyway;
3. whole blocks, four at a time;
4. system RAM.

with `--n-cpu-moe <block_count>` standing in for step 2 on a build that has the
new flag and not the old one.

**A driver that refuses is evidence.** Everything above is planning, and
planning cannot see fragmentation, another process, or a Windows allocator that
will not hand out 17.8 GB in one piece from 22.8 GB free. So a start that ran
out of memory does not merely re-derive the same number: `_next_expert_floor`
raises the floor by two blocks and the retry negotiates against it, alongside
the headroom penalty that has always been applied. The server signature gained
the expert split for the same reason — it is a start-time argument, and a
signature that could not tell 6 from 8 would have handed back the server that
had just failed.

Both flags are capability-gated, separately, because they were added to
llama.cpp at different times and a build may have either. A partial split is
never promoted to `--cpu-moe` on a build that lacks `--n-cpu-moe`: the placement
was negotiated against an estimate of N layers, and moving every expert instead
is a different footprint and a different speed than the one that was planned.
`--flash-attn` and `--swa-full` are unchanged, and MTP is deliberately not
attached to any of these three — the publisher's MTP head is built for a
separate QAT release, and mixing artifacts across releases on the strength of a
shared family name is how a bundle verifies and then fails at load.

### 27.3 A speed is a fact about a placement

`llm:write:<backbone>` was already an improvement on one global number: measured
on one machine in system RAM, a dense 12B wrote at 4.9 tokens a second and the
26B at 12.8, and the catalogue had been implying the opposite from the file
sizes. But the backbone is not the axis that moves it furthest. The same 26B
writes at forty tokens a second resident on a card and at five from system RAM,
and the new rungs in between are real speeds rather than an interpolation of
those two.

So the key gained the placement: `llm:write:<backbone>:gpu`,
`:ncmoe-8`, `:cpu-moe`, `:cpu`, with a `-l20` suffix for the case the design
intent does not list and the ladder can still reach — every expert gone *and*
blocks leaving, which covers placements an order of magnitude apart. Nothing
writes the backbone-wide key any more, so nothing can average two placements
into one number; it is still *read*, last, because a store written before this
change holds it and a stale approximation is a better first guess than none.

The catalogue asks a different question from the progress bar — it is drawn for
entries that are not loaded and mostly never have been at the placement they
would get next — so it goes through `best_measured`, which prefers the running
placement when the entry is the running backbone and otherwise reports the
fastest on record. Either way the line names it: *measured here: 31.4 tokens/s
(8 expert layers in RAM)*, rather than a number that reads as a property of the
model.

## 28. Qwen 3.8, and an accelerator that is not a memory setting (27 August 2026)

Design intent: `DESIGN_INTENT_QWEN38_DFLASH2.txt`, with the developer package
beside it (`IMPLEMENTATION_MAP`, `VRAM_AND_PLACEMENT_MATRIX`, `TEST_PLAN`).

The feature is one backbone family and one accelerator, and the interesting
part of it is a shape decision that the obvious implementation gets wrong.

### 28.1 The switch that would have cost people VRAM

The natural way to ship speculative decoding is one control — call it
*Lightning* — meaning **use the fast decoder and empty the card for it**. It
reads well and it is two facts stapled together:

* an accelerator is a *decoding mechanism*. MTP and DFlash2 produce the same
  tokens as ordinary decoding and produce more of them per step. Neither is a
  quality setting and neither is a memory setting.
* a memory priority is a statement about *ownership of one card*.

Stapled, they produce a machine whose backbone and draft model already fit
having to evict a checkpoint to get a decoder it could have had for nothing.
And on a two-card machine they produce something worse: a Creative Writer on a
3090 selecting Lightning and emptying a 5090 it was never going to be placed
on, which frees not one byte of the card that is short.

So `mc_llm_accel` holds two settings —

    accelerator      auto | none | mtp | dflash2
    memory priority  cooperative | llm_priority

— and the presets are a mapping over them, in `PRESET_AXES`, computed in one
direction and reversed rather than written down twice. `Settings.preset` is
*derived*: the advanced controls are authoritative, and a stored preset that
disagrees with them is a stale label. The combination no preset names is the
one the design intent calls mandatory and is the reason for all of this:

> **DFlash2 + cooperative** — the fast decoder, with nothing released to get
> it, whenever the complete plan already fits.

`custom` is a state the panel *reports* and not one it offers, because picking
it from a menu would mean nothing: the values it would apply are the ones
already in force.

The defaults are `auto` and `cooperative`, which is exactly what every earlier
build did. An upgrade does not begin releasing image VRAM because a feature
exists.

### 28.2 Two stages, because they cost different things

`Runtime.client` reaches the accelerator twice.

`accelerator_choice` asks about *files and records*: does the catalogue entry
advertise anything, is the draft on disk, is a DFlash2 runtime installed, did
it pass its smoke test. Nothing in it reads free VRAM, so its answer does not
change between two requests a second apart — which is what lets it into the
warm server's identity. A forced request that fails here fails before a
twenty-second model load, with a sentence naming the missing piece.

`accelerator_plan` is the fit, and it is the only place image residency can be
released for the language model. It runs on a path that is really going to
start something, for the reason `client` stopped re-negotiating placements on
warm turns: a number that is different every time it is read restarts a server
every time it is read.

The split shows up again in the warm comparison. `_identity` carries the
*chosen* axes — a setting change restarts the server — and `Runtime._accelerator`
carries the *resolved* plan beside it, because `auto` legitimately turns from
ordinary decoding into DFlash2 the moment a draft model appears on disk, with
no setting having changed at all.

### 28.3 There is no smaller DFlash2

`negotiate` exists to make a model fit by giving things up: context first, then
experts, then whole blocks, then system RAM. Every one of those rungs is
forbidden under DFlash2, and not as a matter of taste — a Lightning run that
quietly became a partial offload is *slower than the Normal run it replaced*
and says the opposite on screen.

So a speculative plan does not go through the ladder at all.
`_speculative_negotiation` builds the requested placement and nothing else, and
`accelerator_plan` has already proved the complete thing fits:

    target weights + target cache + recurrent state
  + draft weights + draft cache + compute
  + projector allowance, when *this request* carries an image
  + safety margin

against what is spendable on **one** CUDA card. If it does not fit, the plan is
refused with both numbers in the sentence and a note saying whether any image
VRAM was released. The retry loop in `client` is also cut off for a speculative
plan: every attempt in it asks for less of the card, which is precisely the
outcome that must not be labelled Lightning.

### 28.4 Reclaim, scoped to a card

`mc_broker._victim_order` returns nothing for an LLM request, and that stays
true. `release_for_llm` is a second door, reached only from a plan whose memory
priority the user set, and it is narrower than `request_vram` in three ways
that are the whole point: it is scoped to one physical card, it asks for the
deficit and never sweeps, and it **re-measures** afterwards so the launch
decides against what the driver says rather than against what it hoped.

`_same_card_as_the_image_side` answers *no* to an unanswerable question, which
is the opposite of `mc_llm_runtime.shares_the_image_card` and is the same
caution pointing the other way: there, an unknown card costs a smaller language
model; here it would cost an evicted checkpoint.

One asymmetry is deliberate and worth naming. Under LLM priority, `_spendable`
is asked with `image_budget=False`, so `mc_plan`'s reservation for the image
side stops capping the fit — overriding that reservation on one card is the
entire content of what the user asked for. The *learned* cap is not lifted with
it: that one records an allocation a driver actually refused, and no amount of
permission makes a refused allocation succeed.

### 28.5 A hybrid model keeps memory in two places

Qwen 3.8 interleaves Gated DeltaNet blocks with periodic full attention.
`mc_gguf` already read `attention.head_count_kv` as a per-block array — llama.cpp
writes a zero for every block that keeps no cache — so the cache half was
already right. The other half was missing entirely: those blocks hold a fixed
recurrent state per sequence, and a planner that charged nothing for it reports
a 27B as cheaper than it is, in the direction that ends in an allocation
failure rather than an avoidable eviction.

`Gguf.recurrent_state_elements` is llama.cpp's own arithmetic in its own terms —
a convolution window of `conv_kernel - 1` over the inner width plus the two
group projections, plus `state_size × inner_size` — and `recurrent_bytes` is a
separate term in `Estimate` rather than folded into the cache, because it does
**not** grow with the context and putting it in the cache term would make it
scale with a number it does not follow. A header that does not describe the
shape is charged `RECURRENT_FALLBACK_BYTES` per block instead of zero, and the
measured footprint recorded after a real load supersedes it.

### 28.6 Help text is not proof

Upstream llama.cpp already carries DFlash terminology, so a build that has
never heard of pull request 27342 can advertise `--spec-type` and print
`draft-dflash` in its help. `mc_llm_dflash` therefore treats the help text as a
*necessary* condition asked first because it is free, and writes the capability
record only from a real load of the real target with the real Blackfrost
sidecar answering a question with one right answer.

`dflash2_text` and `dflash2_vision` are two fields because the pull request's
multimodal work moved separately from its text path.

That is also the one place a *forced* accelerator does not refuse. A request
carrying an image, on a runtime whose text path is verified and whose image
path is not, gets a note and ordinary decoding. The distinction is between a
configuration that cannot do what was asked — no runtime, no sidecar, no room,
all of which stop the start — and a *request* this configuration cannot
accelerate, where the very next text request runs on DFlash2 unchanged.
Refusing would also break the operation this product is mostly for: Krea
captions references and then writes a prompt, so an installation with text
verified and vision not — the expected state until phase 3 — would fail every
generation carrying a reference. Section 13 of the intent asks for one
validated mechanism per request, and that is this. It is reported rather than
silent: a warning in the log, a note on the report, and the status line naming
ordinary decoding.

Writing a text result always clears the vision one — in both directions, because a fresh verification
that never sent an image must not leave "vision verified" on screen. The record
carries the executable's size and mtime, and a record whose fingerprint no
longer matches is treated as absent, so dropping a different build into the
same directory does not inherit the previous one's proof.

The family lives under the same `runtime*` naming as every other, because it is
installed by the same staged-then-swapped mechanism — `stage_build`, `swap_in`
and `server_in` were named for it. It is excluded from `runtime_families` and
`detect` by its provenance marker, so a machine whose ordinary runtime is
missing cannot silently adopt an unmerged pull request as the runtime for every
model, every role and every mode.

There is no published archive to download. `dflash2-runtimes.json` describes
the two builds with a null URL and a null digest, and `download` refuses an
unpublished entry rather than guessing at one — there is no route through the
module that fetches bytes this repository does not name. The route that works
today is `adopt`: build the branch at the pinned commit, point Setup at the
directory, verify it.

### 28.7 The sidecar is its own transaction

A managed download is finished by a directory rename, which is why its failure
modes are boring. Adding a 3.86 GB draft to a bundle that exists and is in use
has no directory to rename, so `install_draft` does the smallest thing that
cannot half-succeed: stage and verify exactly as any other artifact,
`os.replace` into the bundle, and **only then** rewrite `installed.json`.

A crash before the rewrite leaves a bundle that does not know about a file,
which `installed` reads as a bundle with no sidecar — which is what it had a
minute earlier. A crash after it leaves a complete one. There is no ordering
that leaves a bundle claiming a draft it has not got, and a manifest write that
fails takes the file back out rather than leaving one.

`Installed.matches` deliberately does not look at the draft: an absent sidecar
is the normal state of a current bundle, and asking for it there would report
every ordinary install as superseded. `drafts()` is the separate question, and
it is false both for "no sidecar" and for "a sidecar the catalogue has moved
off" — the second being a bundle that would start DFlash2 against a draft
nobody tested with these weights.

The staging directory is `<id>~draft`, and `~` is a character `_ID` does not
allow. Two directories rather than one because `_prepare_staging` discards a
staging directory whose expectations have changed, and sharing one would have
fetching a 3.86 GB sidecar throw away nine tenths of an interrupted 22 GB
download.

### 28.8 What the numbers are keyed by

Section 17 of the intent asks for learned speed keyed by backbone, quantisation,
physical GPU, placement **and** accelerator, and forbids averaging a DFlash rate
with an ordinary one. `measurement_token` composes the last three onto the
placement token, omitting each suffix when it would say nothing — so ordinary
decoding on an unknown card is still `gpu`, exactly as it was, and every rate a
machine has already measured keeps answering.

`read_speculation` parses the drafted and accepted counters out of the log,
accepting both shapes llama.cpp has printed them in, and reports `known=False`
rather than zero when it recognises neither: *no drafted tokens were accepted*
and *this build does not report acceptance* are different news. Nothing about a
generation depends on it — parsing a log is the only way to get the figure from
another process, and a parser that failed loudly when a format moved would
break generation for a statistic.

### 28.9 Advertising an option is not accepting a value for it

Reported from a real machine within hours of the first build, and worth the
section because the mistake is easy to make twice.

`runtime_supports` asks whether `llama-server --help` lists a long option, and
that gate has been right for every flag this module added before now:
`--n-cpu-moe` either exists or it does not. `--spec-type` is a different shape.
It has existed since the speculative framework landed, so the gate says yes on
every build in the wild; *which types* it takes is an enumeration that has
grown release by release, and it is printed in the option's own usage line:

```
--spec-type none,draft-simple,draft-eagle3,draft-mtp,draft-dflash,draft-dspark,…
```

The value written here was `mtp`. llama.cpp's is `draft-mtp`. So b10621 printed
`error while handling argument "--spec-type": unknown speculative type: mtp`
and exited before it loaded a tensor — and because that happened inside
`_start_and_smoke_test`, the managed switch rolled back and told the user
*"Qwen 3.8 27B Abliterated — Medium was downloaded but would not start"*, about
a backbone that was perfectly fine.

Three things came out of it.

**The spelling.** Both constants are prefixed now, `draft-mtp` and
`draft-dflash`, and the docstring beside them says why the prefix is worth a
sentence.

**A second gate.** `runtime_accepts(flag, value)` asks whether the value
appears as a whole word anywhere in the help output, and both flag builders now
take an `accepts` predicate beside `supports`. The match is deliberately loose
rather than a parse of the usage block — llama.cpp has formatted that block
three ways and `draft-dflash` is distinctive enough that finding it anywhere
means the build knows it. A false negative costs an accelerator that would have
worked, and says so; a false positive costs one failed start, which is what the
third thing is for.

**The model stops paying for this extension's mistakes.** `read_failure` now
recognises `error while handling argument`, and `Runtime.client` responds to one
by dropping the accelerator and starting again — once — rather than by letting
the failure stand. The scoping is the careful part in two directions:

- Only flags on `OPTIONAL_FLAGS` — the ones this extension *chooses* to append
  — are treated this way. `--device` produces the identical message shape and
  is a symptom of a card that could not be enumerated, which is diagnosed above
  in far more useful terms; the existing tests for that caught the first
  version of this branch swallowing it, which is exactly what they are for.
- Once. A second argument error is a real failure, and retrying past it would
  hide whatever is actually wrong behind a loop.

The refused value goes into a session-scoped negative cache, so the next start
does not repeat the doomed attempt. Not persisted: relearning it costs one
failed start per session, and a file on disk claiming a runtime cannot do
something would outlive the runtime being replaced.

This is also why the DFlash2 capability record exists in the form it does. The
help text is evidence, and it is evidence about what a build *says*. Only a
real load of the real target answering a real question is evidence about what
it does.

### 28.10 What is not here

**No published DFlash2 archive.** The manifest entries carry the shape, the
commit, the CUDA version and the compute architectures, and null bytes. Filling
them in is building the branch, hashing the archive and editing one file; until
somebody does, the `adopt` route is the whole of it and the panel says so.

**No cross-device split.** llama.cpp exposes draft-device controls and v1 does
not use them. Target and draft go on one card or the plan is refused: splitting
them complicates the performance contract for a path this product does not
need.

**No Q8_0 tier.** It is about 29 GB, which leaves a 32 GB card too little for a
draft model, 8K of state, compute buffers and a projector. Q6_K is the highest
sensible managed weight and Q5_K_M is the recommended one, and promoting Q6
past it is a decision for the project's own prompt-quality corpus rather than
for a file size.

## 29. Three reports from one evening (27 August 2026)

Unrelated to each other except in having been found in the same session, and
recorded together because two of them are the same mistake in different
clothes: a control that reads a value from somewhere it has no business reading
it from.

### 29.1 New + Save renamed the character you had

Reported as *"I created a new character, tried to switch back to existing and
it wasn't an option."*

`save_character.click` was wired `inputs=[character, name, …]`, where
`character` is the **Talking to** drop-down, and `_save_character` passed that
first value to `CharacterStore.save` as `previous_name`. That parameter means
"the character this file is currently on disk as", and `save` acts on it:

```python
renaming = bool(previous_name) and previous_name != character.name
...
old.unlink(missing_ok=True)
```

So pressing New, typing a name and pressing Save renamed whichever character
was selected — moved its file, carried its picture across with it, and deleted
the original. The list then held one character where there had been two, and
the one the user started from was gone. Not a display bug: their file was
unlinked.

The drop-down says *who you are talking to*. It has never said *what the editor
is editing*, and the fix is to stop pretending it does. `editing` is a `State`
holding the name the editor is bound to — the empty string while a new
character is being written — and Save reads that and nothing else. Around it:

- `_editor_fields` is the single shape every handler touching the editor
  returns, so none of them can leave the name box, the sampling and the
  character being written to describing three different characters. A test
  asserts all five answer the same length, because a handler returning a
  different number of values is a panel that breaks on a button press.
- **New** resets the Advanced sampling with the rest of the editor. Leaving it
  alone had a new character silently inherit the settings — including the
  seed — of whichever one was selected when New was pressed.
- **Cancel** restores those boxes from the selected character, because they are
  also the *conversation's* per-message override and New had just cleared them.
- Creating over a name already taken is refused. Overwriting somebody's
  character silently is the same loss by a shorter road.
- Saving sets `editing` to the saved name, so a second Save edits rather than
  trying to create again.
- **↻ Refresh** re-reads the folder, and **Delete** lands on whatever is left
  rather than on nothing.

### 29.2 The path the panel already knew

Reported as *"Even though I chose to download the DFlash2 llama.cpp build, LLM
Studio doesn't seem to know its location."* — and the clarification is the
whole of it: the button pressed was *Download the draft model*.

Under one Advanced heading sat a download button for the **draft model**
(weights, in the backbone's bundle) and a path box for the **llama.cpp build**
(a program, in its own directory). Two installs, one heading, and a download
next to the wrong box. Pressing the download and then the button beside the
empty box produced *"Enter the path to the DFlash2 llama.cpp build"* — true,
unhelpful, and silent about the fact that the thing being asked for has no
download anywhere yet.

Three changes, and only one of them is code that does anything:

- The two are **numbered and separated on screen**, each with a sentence saying
  it is needed as well as the other and not instead of it.
- The runtime has a **Download button of its own**, wired to
  `mc_llm_dflash.download`. There is no published archive, so what it does
  today is say so and name the commit to build — which is the answer to the
  question that was actually being asked. A button that answers is worth more
  than no button at all.
- The **path is kept**. The box is filled from `mc_llm_dflash.installed()`, an
  empty box means "use the build you have" rather than an error, and adopting
  rewrites the box to where the build now *lives* — inside the data directory
  rather than wherever it was compiled — so pressing the button twice is a
  no-op instead of a second install of the same thing.

### 29.3 A seed of 7

`prompt_engine.options.DEFAULTS["seed"]` is `7`, and Prompt Studio's box opened
on it through `initial("seed", 7)`. That value is not a default anybody chose
for a user: it is what upstream's node self-tests use, picked so two runs come
out identical, which is the exact opposite of what a seed control is for. A
panel that opens on it hands every user a generator that repeats itself until
they notice the box.

So the seed is the one control in that panel that does not go through
`initial`: it reads `stored` directly and falls back to `RANDOM_SEED`, which
leaves a seed somebody actually chose still remembered and still winning.
MiniMax and Krea already opened on −1 and now say so with the constant rather
than a literal — the same reason the Conversation panel gives about its
Advanced accordion, that a second set of literals in the UI is how a panel
quietly stops matching the engine behind it.

Conversation was already right in the two places that matter: `Character.seed`
defaults to `RANDOM_SEED`, and the box does too. What it did not do was reset
that box when New was pressed, which is 29.1's bug wearing this one's clothes.

## 30. DFlash2 taken back out (27 August 2026)

Reported after one evening with it: *"DFlash mode is just not available for my
CUDA, please remove it and LIGHTNING FAST mode."*

It is worth recording as its own section rather than quietly reverting, because
the thing that was wrong with it was not a bug. Every mechanism in section 28
worked as designed: the runtime family installed apart, the capability record
refused to trust help text, the planner would not report a partial offload as
Lightning. What it could not do was run, on the machine it was built for,
because DFlash2 is llama.cpp pull request 27342 and that user's CUDA build is
not it — and no amount of correct refusal machinery is worth carrying for a
mechanism that never starts.

So the accelerator, the preset, the runtime family, the capability record, the
speculative sidecar, the full-residency planner and the two Setup rows are
gone: `mc_llm_dflash.py` and `dflash2-runtimes.json` deleted,
`accelerators.dflash2` out of the registry, `Speculator` and the draft out of
`mc_llm_managed_models`, and `Plan` down to the mechanism and the binary.

### 30.1 What stayed, and why it is not sentiment

**The two axes.** Acceleration and memory priority are still two settings and
not one switch. That was never a fact about DFlash2 — it is a fact about the
difference between *how a model decodes* and *who owns the card*, and MTP with
cooperative memory is as real a combination as DFlash2 with cooperative memory
ever was. It is still the one no preset offers and still reachable in one
dropdown.

**The value gate.** `runtime_accepts` and the argument-error retry came out of
section 29.3 and apply to `draft-mtp` exactly as they did to `draft-dflash`.

**The hybrid state arithmetic.** Qwen 3.8 interleaves Gated DeltaNet blocks with
periodic full attention whatever is decoding it, and a planner that charges
nothing for the recurrent half reports the model as cheaper than it is.

**The two-stage choice.** `accelerator_choice` still answers from files and
`accelerator_plan` still asks the binary, because the first is stable enough to
put in the warm identity and the second costs a `--help`.

### 30.2 The setting that would have become a lie

`llm_priority` had exactly one caller: the DFlash2 fit. Removing DFlash2 without
noticing would have left **Fast LLM** as a preset whose second half did nothing
at all — a control that says "this card's image residency may be released for
it" and never releases anything.

So `_make_room_for_the_llm` now runs on the ordinary placement path: under LLM
priority, and only then, it asks the broker for the configured placement's
deficit on the card that placement is going to. It sits in `Runtime.client`
before the negotiation rather than inside it, because `negotiate` promises to
move nothing and the estimator relies on that — what this does is find more
free VRAM than there would have been a moment ago and let the ordinary ladder
place against it.

`mc_broker.release_for_llm` is unchanged and is still the narrow door section
28.4 describes: one card, the deficit, re-measured, never a sweep.

### 30.3 What an upgraded installation is left holding

A `draft.gguf` inside any bundle whose sidecar was downloaded — about 3.9 GB,
now referenced by nothing. It is **not** deleted on upgrade: reading a manifest
is not a licence to remove four gigabytes of somebody's disk, and a bundle with
an unreferenced file in it is a bundle that works. `cleanup` sweeps the
`~draft` staging directory, which nothing writes any more; the installed file
is the user's to remove.

## 31. Five things after an evening in Conversation (27 August 2026)

Reported together, and they are five separate things: one is a resource leak,
one is a layout, two are the same complaint about regenerating, and one is a
thing the panel knew and would not say.

### 31.1 A server that outlived the WebUI

> *"If I kill the webui process, there tends to be llama-server.exe running on
> my system. Can these processes be closed on webui exit?"*

Yes, and this is the failure that costs the most: a resident model is several
gigabytes of VRAM that nothing on the machine will reclaim, and nothing tells
you it is still there. There was already a stray sweep — `release_strays`, run
when the extension loads — but sweeping on the way *in* only ever helps the next
session, and only if there is one.

Three doors, because a WebUI stops in three different ways and only the first of
them runs any Python of ours:

- **`on_script_unloaded`** — the host's own "the UI is going away" callback, in
  `scripts/model_chain.py`. `shutdown()` stops every runtime in the registry and
  then calls `release_strays()`, so a server whose `Popen` handle was lost to a
  reload is stopped as well as the ones still held.
- **`atexit` and the signals.** `stop_on_exit()` arms an `atexit` hook and
  chains `SIGTERM` and `SIGINT` — *chains*, so the previous handler still runs
  and Ctrl+C in the console still stops the WebUI the way it always did. It is
  armed from `_repair_launcher()`, which is to say the first time this extension
  is in a position to have started anything.
- **The job object**, for the case none of the above covers: an End Task, a
  crash, a kill -9. On Windows every server is created inside a job created with
  `JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE`, and the kernel ends the job — and every
  process in it — when the last handle to it closes, which happens when the
  WebUI process dies however it dies. `_die_with_us` assigns each child as it is
  launched, from `_Launcher.Popen`, and fails quietly on anything that is not
  Windows or refuses the assignment, because a server that has to be stopped by
  one of the two doors above is still better than no server at all.

### 31.2 A header that scrolled away

> *"I would like the options bar to stay docked to the top of the screen in
> conversation mode."*

`position: sticky; top: 0` on `.mc-llm-chat-header`, with the surface's own
background painted behind it and a `z-index` above the transcript. It needs the
background stated: a sticky element with a transparent background has the
messages scrolling *through* it.

### 31.3 Regenerate, twice

Two of the five were the same feature, and the second is the one that mattered.

> *"Add a regenerate icon directly next to the chat response in thread view."*

There is nowhere in a Gradio 4.40 `Chatbot` bubble to put a component, so the
icon is drawn in `javascript/llm_studio.js` and does exactly one thing: it
writes **which reply it is on** — counted down the transcript — into a hidden
`Textbox` and presses a hidden `Button`. `_regenerate_reply` turns that ordinal
into a message index through the same `positions` map a click on a bubble goes
through, and then calls `_regenerate`. The browser nominates and decides
nothing, which is the line that file stays on.

The ordinal is read **when the icon is tapped**, not when it was drawn. An
ordinal captured in a closure names the wrong message as soon as anything above
it is deleted, and for this button naming the wrong message means rewriting a
reply the reader did not point at. There is a node test that moves the bubbles
underneath the icon and then taps it.

This is also the one place in that file that reads the host's own DOM, because a
bubble carries no id of ours. The shapes Gradio 4 and the themes that reskin it
are known to use are tried in turn, and when none matches **no icon is drawn and
nothing else changes** — Regenerate is still on the sheet a tap on the bubble
opens. A theme this cannot read costs an icon, never an action.

### 31.4 What regenerating in the middle used to cost

> *"When I decide to regenerate a message that is mid threaded response, it
> should effectively create a branch. … If I go back to the original response, I
> expect the entire thread to load!"*

It did not, and the reason is one line:

```python
conversation.truncate_after(index)
message.add_version("")
```

Regenerating **deleted every message after the reply** and then added a version
to it. So `◀ 1/2 ▶` paged back to the original reply with the rest of the
conversation gone for ever — the versions were the only thing kept, and a
version is one string.

That is the shape of the bug, and it says what the fix has to be. The thing that
has to come back is *every message that followed*, and a list of strings on one
message cannot hold it. The store's own word for a conversation that diverges is
a **branch**, and `ChatStore.branch` already made one.

So `_regenerate` is now positional:

- **At the end of a thread**, where nothing follows, it appends a version and
  `◀ 2/3 ▶` pages between the attempts. Nothing is lost because there was
  nothing after it, and that is what makes it reversible.
- **Anywhere else**, it branches at the message *before* the reply — so the
  branch ends on the turn the reply was answering — writes the new reply there,
  and moves the panel onto it. The thread it came from keeps every word.

Index nought is included in "anywhere else", and it is the case that would have
been easiest to leave out: an opening reply has no turn before it, so the branch
starts empty. Still a branch, and still nothing lost, which is the whole point.

The two rules together are also what makes the pager safe from here on: versions
now only ever exist on a message with nothing after it.

Regenerate is the one streaming handler whose outputs carry the thread list and
the open thread, because it is the one that can change which thread the panel is
on — `_into_thread` prefixes every event `_stream` yields. A panel left pointing
at the thread it came from would apply the next action to the wrong
conversation.

### 31.5 A system prompt the panel would not show

> *"Please expose the current system prompt in the character view. … Let me see
> the default right in the character edit view so I can edit if I want."*

The character editor now shows the prompt **composed the way it will be sent**,
by calling `prompt_master.chat.prompt.system_text` — not a description of it and
not the template with the fields left in. It follows the name, the description
and the override as they are typed, so what a change does is visible before it
is saved.

**Edit this system prompt** copies what is showing into the override box, where
it becomes this character's own; emptying that box puts them back on the common
default. It will not overwrite an override somebody has already written, and
what it copies is what was *running* — the preview and the copy are the same
call, so the two cannot disagree.

The preview never raises into the panel: a character mid-edit is a character
that may not compose, and a stack trace where a prompt should be is worse than a
sentence saying so.

## 32. A prompt is edited where prompts are written (27 August 2026)

Two more after another evening, and the second one is the answer to a thing 31.2
did not actually fix.

### 32.1 Edit was two verbs wearing one word

> *"When I decide to edit a user submitted prompt in conversation mode, the edit
> view is messed up. It seems to hang."*

Underneath the symptom is a category error, and it is worth naming because the
symptom is only what it looked like from outside.

**A reply is text on the transcript.** Editing it means rewriting that text, and
an in-place editor is the right shape for that.

**One of your own messages is a prompt that was sent.** Editing it means sending
a different one — and the answer it already got is part of what is being
replaced. Rewriting the string in place produces a thread where you asked one
thing and were answered another, with no way to re-ask short of finding
*Send again from here* on a sheet.

The old editor treated both as the first thing: a second text box under the
transcript, with its own Save, borrowing the composer's space. So editing a
prompt left the panel showing an editor where the composer had been, and a
message changed but never re-asked. The rule that replaces it was given with the
report:

> *"If I want to edit a prompt, I should see it immediately in the user prompt
> field, where if I submit a prompt mid thread (where there are replies after)
> it should start a branch."*

So a message of yours is **taken back** — `_take_back`: lifted out of the thread
and into the composer, where it is an ordinary unsent message and Send is
already the thing that sends it. No new state, no second box, no third button.
The transcript and the composer never hold the same message at once, which is
the property that makes it legible: what is in the box is what will be sent, and
what is in the thread is what was.

Mid-thread, lifting it would destroy every reply that followed, so it happens
**in a branch** — the thread is copied up to the turn before the message, and
that copy is what gets edited. Section 31.4's rule, applied to the other half of
the transcript, for the same reason.

The branch is made when Edit is pressed rather than when Send is. The end state
is the same either way, and doing it early means the panel can *show* you that
you are in a new thread before you commit to anything, rather than moving the
ground after you press Send. It also means no state has to be carried between
the two presses, and stale state is what the first version of the character menu
lost people's files to.

Two edges worth stating:

- **The first message of all** has no turn before it, so the branch starts
  empty. Still a branch; the thread it came from is still whole.
- **A picture cannot come back.** The composer's attachment is a file on disk
  and a saved one is a data URL inside the thread. Rather than invent a
  temporary file for the rarest path here, it is said out loud — and in the
  branching case nothing is lost at all, because the thread it came from still
  has it.

Saving an edited **reply** now returns to the conversation rather than reopening
the action sheet over the message it just saved. The sheet covers the bottom of
the transcript; putting it back up after the work is done is the panel looking
stuck on a finished thing, which is the other half of what "hangs" described.

### 32.2 A message nobody ever answered

The same lift, applied automatically. A thread can end in one of your messages
with no reply: `_send` saves your message before the request goes out —
deliberately, so it survives the reply not arriving — and `_tidy` then removes
the empty bubble that was going to hold the answer. What is left is a question
nobody answered, and the only thing to do with it is ask again.

So `_unanswered` finds exactly that shape and `_open_thread` puts it back in the
composer. Guarded twice: never over a box with something in it (the message
stays in the thread, where Edit will still take it back later), and never one
carrying a picture.

### 32.3 The footer, and why 31.2 did not dock anything

> *"#3 from the last improvements did not work, the bar is not docked… The real
> problem is that in forge neo web ui, the FOOTER takes up real space on the
> page."*

Correct, and the diagnosis is the fix. `position: sticky` sticks an element
within its **scrolling ancestor**, and the conversation header's scrolling
ancestor is the workspace — a flex column of a fixed height, in which the header
is already `flex: 0 0 auto` and already stays put while the transcript scrolls
under it. The header was never the thing moving. *The page* was moving, and
taking the whole workspace with it.

The workspace is measured to end 16px above the bottom of the viewport, so
nothing below it should exist to scroll to. One thing does: the WebUI's footer,
which is outside anything this extension lays out and adds a strip below the
fold. The page then scrolls by the height of a row of links, and everything —
header included — goes up with it.

`model_chain_hide_footer`, on by default, takes it off the page. The shape is
the one the progress styling already uses and needs no endpoint and no Gradio
component: every registered option is dumped into the browser's global `opts`,
`llm_studio.js` reads it and writes `data-mc-footer="hidden"` on the root, and
`style.css` does the hiding. So it is one attribute, overridable by a theme or a
user stylesheet, nothing is removed from the DOM, and turning the setting off
puts the footer back on the next update rather than at the next reload.

It applies to every tab, which is stated in the setting rather than hidden: the
footer is one element on the page and there is no per-tab version of it.

The sticky rule stays. It does nothing in the ordinary layout and costs nothing,
and it is the correct behaviour for the case where the workspace itself is the
thing that scrolls — a window too short to lay out in, where `fit` publishes
nothing and `style.css` hands the page its scroll bar back.

### 32.4 One try/catch was one too few

Found while testing the above, and worth its own note because the failure was
silent. `wire()` ran all six features inside a single `try`, so the first one
throwing took the five after it with it — a tab with no reply icons, no
Ctrl+Enter and no fitted workspace, and one line in the console. The file's own
header promises the opposite: *"if an id is missing, the feature it drives is
skipped and the rest carry on."*

Each concern is now wired through `attempt(what, run)` with its own `try`, and a
node test breaks the first feature deliberately and asserts the icons are still
drawn.

## 33. Vision was a mode, and it should have been a capability (27 August 2026)

Local multimodal inference did not work, and the reason was not one function. It
was a lifecycle mistake repeated in three places, each of which was individually
defensible.

### 33.1 The three facts that had become one

There are three separate things to know about a multimodal backbone, and the old
code kept them in one variable:

| | |
| --- | --- |
| **declared** | the catalogue says this backbone has a projector |
| **present** | that projector is a file on this disk |
| **loaded** | the running llama-server was started with `--mmproj` |

Conflating them produced two bugs pointing in opposite directions.

*The selection bug.* The ordinary model chooser records a model with no
projector — correctly, because a filename does not prove which weights a
projector was made for — and clears `mmproj` doing it. Recognising the path as a
managed bundle afterwards restored the source, the id and the hidden profile,
and not the projector. What that left was a state file which knew a managed
multimodal backbone was selected and reported that it had no eyes: *declared*
was true, and the only record of it had been erased. `follow_path` now restores
the association with the profile, and the switch path records the declared path
rather than the present file, because a bundle whose projector has been deleted
still has one declared for it and that is what a repair works from.

*The lifecycle bug.* `Runtime.client` read the current request's `needs_vision`
as a complete description of the server that ought to be running:

```python
projector = configuration.mmproj if needs_vision else None
```

So `needs_vision=False` did not mean "no upgrade required". It meant "restart
without the projector". A conversation with pictures in it thrashed:

```
    text  ->  text server
    image ->  restart with the projector
    text  ->  restart without it
    image ->  restart with it again
```

Every one of those restarts costs a model load, a CUDA context, a VRAM
re-placement and llama.cpp's prompt cache — which is the whole conversation
re-evaluated from the first token before a single character appears.

### 33.2 Capability, and it only moves upward

The runtime now records what the *process* was started with, in `_projector`,
beside the identity that records what the *user* chose. A request asks for the
capability it needs rather than for the server it imagines:

```
    OFF  ->  TEXT_ONLY  ->  VISION_LOADED
```

`_wanted_projector` is the whole rule. A picture names the compatible projector.
Anything else inherits whatever the running process already holds, because
vision capability is a superset of text capability and a superset satisfies the
subset. There is no path by which an ordinary request moves the state back.

Two things had to follow it or the fix would have been undone elsewhere:

- **`_outgrown` previews with the projector accounted for.** A vision-loaded
  server is holding over a gigabyte, so a placement previewed *without* it looks
  roomier — and "it now fits better" is a restart. That would have reintroduced
  the thrash through the placement path instead of the capability path.
- **`_restart_in_system_ram` carries the loaded projector, not the configured
  one.** It is a relocation of the server that is running, so it has to come
  back with the capability it went away with. Starting from
  `configuration.mmproj` would have given a text-only server a projector in
  system RAM to satisfy nothing; starting from `None` would have taken vision
  away from a conversation that was mid-picture.

Stickiness ends where the process does: Unload, a model or device change, a
broker eviction, a crash, or shutdown. It is per runtime, so one server
acquiring a projector leaves every other server's process, placement and prompt
cache untouched.

### 33.3 Repair happens before the lock, not inside it

`mc_llm_vision` is the new module, and it is small on purpose: it answers "which
file" and leaves "is it loaded" to the runtime. When a managed bundle's
projector is missing, the first image request downloads the exact catalogue
artifact — same revision, same SHA-256, through the same verified downloader —
and records it, without anybody being asked to go and find a file.

`repair_projector` is deliberately not `download()`. That call is a whole-bundle
transaction: stage everything, verify everything, rename a finished directory
over the installed one. Re-running it to obtain one missing sidecar would
re-verify, and possibly re-fetch, seventeen gigabytes of weights that are
already on the disk and quite possibly mmapped by a server that is answering
requests. So one artifact is staged, verified and `os.replace`d in beside
weights nothing touched, and its own staging directory is separate so an
automatic repair cannot discard somebody's paused model download.

It runs *before* `Runtime._lock` is taken. A gigabyte over somebody's connection
must not hold the process lock of a llama-server that is answering other
requests perfectly well — and nothing is stopped at that point either, so a
repair that cannot be completed leaves a running text server running rather than
destroying a usable text model over an optional upgrade. The run's cancellation
is passed down, so Stop stops the transfer and what has arrived is kept.

### 33.4 What "OpenAI-compatible" does not mean

llama.cpp speaks an OpenAI-compatible chat schema. Those words describe the
shape of a JSON body; they authorise nothing. `prompt_master/inference/
local_only.py` turns that from a comment into two checks:

- **The endpoint is loopback or it is refused** — at construction, which is the
  last moment before a prompt, a character card, a Creative brief or Spatial
  data can be attached to it. There is no setting anywhere in these paths that
  can redirect inference at a host, public or private, and no fallback to a
  hosted API on any failure.
- **Images are embedded or they are refused** — llama.cpp will fetch a remote
  `image_url` if it is given one, which would make the *server* perform a
  data-dependent network request. Every caller in this project already builds a
  `data:` URI; the guard is at the point of transmission so that stays true of
  the sixth caller as well as the five that exist.

The only thing that reaches the network is provisioning: a model, a projector, a
runtime binary. With those on disk, pulling the network cable changes nothing
about inference.

### 33.5 What the tests assert

Mostly process identity and counts, because a test that only checked each
request eventually succeeded would have passed against the broken lifecycle. A
mixed session of `text, text, image, text, text, image, text` has to produce
exactly one start and exactly one capability upgrade, and the process object has
to be the same one across every turn after it. Two runtimes are driven side by
side and the second one's process identity is asserted unchanged while the first
upgrades.

The provisioning validator does the half that mocks cannot: it starts the real
llama-server the install has just downloaded, proves the text probe loaded no
projector, upgrades once for an embedded image, and then requires the *same
process* to answer a text follow-up. Upstream multimodal regressions have
shipped in builds where the request schema and the projector configuration were
both correct, and only a real binary can catch that.

### 33.6 The picture never got past Gradio (27 August 2026)

Reported after the above shipped: attaching an image to a conversation still
produced a text-only reply, and the console carried three copies of

```
File "gradio/components/image.py", line 197, in preprocess
File "gradio/image_utils.py", line 30, in format_image
    path = processing_utils.save_pil_to_cache(im, cache_dir=cache_dir, name=name, format=format)
TypeError: save_pil_to_file() got an unexpected keyword argument 'name'
```

Nothing in this extension appears in that traceback, and nothing in it needed
to. Gradio 4.40's `format_image` preprocesses a `type="filepath"` image by
calling `processing_utils.save_pil_to_cache(im, cache_dir=…, name=…, format=…)`.
The WebUI replaces that function — `modules/ui_tempdir`, so that saved images
carry their PNG info — with one whose signature is
`(pil_image, cache_dir=None, format="png")`. It predates the `name` parameter.

So *every* `gr.Image(type="filepath")` input in this host raises inside
`preprocess_data`, before any handler runs. The four image inputs LLM
Studio owns — the Conversation attachment, Prompt Studio's start frame,
MiniMax's reference frame and Krea's four reference slots — were all of them
that kind, which is the whole of "vision does not work" as a user experiences
it. The llama-server log for the reported session confirms it from the other
end: two text prompts, 290 and 377 tokens, no projector, no image.

The fix is not to patch the host. `format_image` returns the decoded image
immediately for `type="pil"` and never reaches the replaced function, so the
four inputs ask for the picture instead of a path. That is also the better
answer independent of the bug: one decode rather than a second copy of
somebody's photograph written into a cache and read straight back, and the
bytes never touch the disk on the way to a local model.

Output is unaffected and was never broken — `save_image`, which postprocesses a
component's value, does not pass `name`.

What it costs is the filename. Gradio consumes the upload's original name
inside its own decode (`im.convert(self.image_mode)` returns a new image, and
`PIL` does not carry `filename` across that), so a handler receiving a picture
receives no name for it. `ui.picked_name` answers `"attached image"` there
rather than an empty string, because empty would take the `*[…]*` marker out of
the transcript and leave nothing on screen to say a picture had been attached
at all. A path still contributes its basename, and never the path itself.

`ui.data_url` now takes either shape, `prompt_master.imaging.preprocess.encode`
is the half that works on an already-decoded picture, and Krea's `Reference`
grew a `picture` field beside `path` so a slot can carry either without the
numbering contract in §4 knowing which. A test asserts no image input in the
extension asks Gradio for a filepath again, and a second one asserts the inputs
still exist — because deleting them is the other way to make the first pass.

## 34. Six things about pictures and chrome (27 August 2026)

### 34.1 A conversation could not keep its pictures

A chat carried its attachments inside itself: the JPEG the model was shown,
base64-encoded, in the same JSON file as the words. Durable, and honest — the
file held exactly what was sent — and wrong in three ways that only appear once
somebody uses it for a week.

*It could not be shown.* The transcript is re-sent to the browser on every token
of a reply, so a picture inside a message is re-sent on every token: a hundred
and fifty kilobytes per chunk, per image. Which is why the transcript only ever
rendered `*[frame.png]*` — a line of italic text where a photograph had been.
Reported as a screenshot of exactly that, a week later, with the reply
underneath about something the reader could no longer see.

*It could not be found.* A picture inside a JSON file is not a picture on a
disk. No folder, nothing to look through, nothing to delete without editing a
conversation by hand.

*It could not be shared between turns.* The same picture attached twice was
stored twice; a thread branched five times was six copies.

So `mc_llm_attachments` puts the bytes on disk and the conversation keeps a
path:

```
<LLM data root>/chat-images/<character>/<content hash>.jpg
```

Content-addressed inside the character's folder. Storing the same picture twice
finds the file already there and writes nothing, a branch shares its parent's
pictures, and an edit that re-attaches the same photograph is a no-op — the name
*is* the hash, so all three fall out of one rule rather than three checks.

The transcript then shows the picture through the host's own `file=` route,
which costs a path per token instead of a photograph. That route needs the file
to be allow-listed; the WebUI launches with its data directory already on
Gradio's list and the LLM data root is inside it, so most installations need
nothing — and where they do, `modules.ui_tempdir.register_tmp_file` is the
host's own answer to the question and is what gets called, memoised, because
`markup()` runs once per message per token.

`Message` grew `image_path` beside `image`, and writes one or the other and
never both — writing the inline copy beside the path would keep every migrated
chat exactly as large as it was, which is the thing moving them out was for.
Chats written before the folder existed are migrated when they are opened,
once, in `_load`, rather than by a script somebody has to know to run. A picture
that cannot be decoded is left exactly where it is: losing an attachment to a
tidying-up would be far worse than a chat that goes on carrying one.

Nothing deletes. A picture is reachable from several threads and from several
branches of one thread, so no thread can know it is the last one holding it. The
folder is the user's, and being able to find and delete these by hand is what it
was asked for.

### 34.2 The attach control was a panel

Pressing the paperclip opened a full-width drop target above the composer:
a panel's worth of empty dashed border to say "no picture yet", on the one
surface in this mode that must never grow. Reported with a screenshot of the
empty gallery stretched across the window.

The picker was always one tap away inside that target, so the paperclip now
forwards its press to the component's own file input —
`javascript/llm_studio.js`, the same place the per-reply regenerate icon lives —
and the component is a 4.5em chip inside the composer row, not in the layout at
all until there is a picture in it. Python still handles the press as well: it
makes the chip visible and says whether the model running can be shown a picture
at all, so a browser where the script did not run has a target to click rather
than nothing.

`sources=["upload"]` and not the default three. With more than one source Gradio
draws a chooser and the file input is not in the DOM until somebody has picked
"upload", which would make the paperclip do nothing.

### 34.3 Scrolling past the bottom of the content

Hiding the WebUI's footer (§32.3) took away the links and left the room they
were in. The page still scrolled, now into blank space.

`fitOne` measures from the top of the workspace to the bottom of the window,
which is right only if nothing sits below the workspace — and something always
does: the container's own bottom padding, and whatever the host puts after the
tab. No list of selectors can be relied on to find every element that could be
responsible, and hunting for them is how this kind of fix breaks under the next
theme.

So none is used. What is measured is the *result*: after the height is applied,
if `documentElement.scrollHeight` still exceeds `clientHeight`, the workspace
gives back exactly that much. Reading `scrollHeight` forces the layout that was
just invalidated, which is the point — the correction is against what the
browser did, not against what the script expected it to do. It converges because
it only ever shrinks and because `documentTop` is scroll-invariant, and it stops
at the floor the layout needs, where the honest outcome is that there is not
enough window and the page keeps its scroll bar.

### 34.4 Three destinations behind a menu

Threads, Character and You were entries in a menu the `☰` opened. A menu whose
entire contents are three destinations is a tap in front of each of them, and
they are the three this mode is navigated by, so they are buttons on the header
now. The answer to a narrow display is `flex-wrap: wrap` — a second line of
chips, which is a smaller loss than a hidden one.

### 34.5 One button, one behaviour

`☰` opened LLM Studio's workspace chooser everywhere except Conversation, where
it opened a menu of Conversation's own. Same glyph, same corner, two meanings.
Conversation's menu is gone, so the button does what it does everywhere else:
the panel's own handler puts its surfaces away and the shell's second handler on
the same button opens the chooser.

The shell's sheets also moved to the left on a wide display. They were anchored
`left: auto` — the far corner from the control that had just been pressed, which
on a 27-inch monitor is most of a screen away from where the eye already is.

### 34.6 Edit had become a second Branch

Editing one of your own messages *took it back*: lifted it out of the thread and
into the composer, where Send would ask it again. Mid-thread that would have
destroyed the replies after it, so it branched first — which meant Edit and
Branch did the same thing, one button apart.

Edit now edits. The words in the thread change, the thread stays where it is,
and the replies stay under it. That the conversation's meaning changes is the
feature and not a side effect: a thread where you asked about the sky, were told
"blue", and then made the question about the sun is a thread you can go on to
ask about. The file is the only record there is, and what it says is what was
said.

Both roles go through one path now, so `SELECTION_ORDER` grew `edit_image` and
the editor gained a picture chip of its own — a message that was sent with a
picture is edited as a whole, so the picture has to be changeable and removable
where the words are. The chip is filled from the message's stored path; the one
case where it is *not* the message's picture is a chat too broken to have been
migrated, and `_commit_edit` checks for that rather than reading an empty chip
as "the user took the picture away".

`_take_back` is gone. `_unanswered` — a message that never got a reply going
back into the composer when the thread is opened — is a different feature that
happened to share it, and is unchanged.
