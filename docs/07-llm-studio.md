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
