# The Forge Assistant — implementation notes, gates and known limits

Written against specification 5.0 of 2026-09-21, which was itself verified
against this repository at `78da206`. This file is the other half of that
document: what was built, what was deliberately not, what the specification got
wrong, and what still has to be checked against a running Forge before the panel
is turned on by default.

It is developer notes, not a second specification. Behaviour that people see is
in `README.md`; the reasoning behind a particular line of code is in that
module's own docstring, which is where it stays true.

---

## 1. What is here

Nine Python modules, four browser files, one stylesheet section, and eight test
files.

| | |
|---|---|
| `mc_llm_conversation_store.py` | locks, revisions, fingerprints, `transaction()`, receipts, tombstones, collision-free names |
| `mc_llm_conversation_service.py` | envelope validation, deduplication, every message action, snapshots, capabilities |
| `mc_llm_conversation_ops.py` | a reply from acceptance to guarded completion; checkpoints, recovery, completion records |
| `mc_llm_conversation_feed.py` | page feeds: monotonic cursors, bounded rings, TTLs |
| `mc_llm_conversation_api.py` | the browser routes, the capability and the event stream |
| `mc_llm_conversation_startup.py` | orphans, interrupted replies and staged uploads, once per process |
| `mc_llm_attachment_staging.py` | uploads validated, decoded and held, with explicit readiness |
| `mc_llm_overlays.py` | one owner for all seven pop surfaces |
| `mc_assistant_settings.py` | the panel's settings, validated in Python as well as in the browser |
| `javascript/forge_assistant.js` | launcher, six-anchor docking, panel, transcript, composer, actions |
| `javascript/forge_assistant_store.js` | the page's copy of the conversation, drafts, operations |
| `javascript/forge_assistant_host.js` | Forge's tabs and header, as an adapter; keyboard arbitration |
| `javascript/forge_assistant_focus.js` | focus as a reversible transaction, with editor adapters |

Changed: `prompt_master/chat/history.py` (a revision, a guarded save path, a
tombstone file, an exclusive identifier), `mc_llm_chat_panel.py` (fifteen direct
writers replaced by commands; `_stream` replaced by a follower; Stop re-pointed;
the process-global completion slot removed), `mc_llm_studio.py` and
`mc_voice_ui.py` (the overlay owner, the mode publish, the operation-scoped
speech marker), `mc_llm_attachments.py` (ticketed serving), `javascript/
voice_chat.js` (origin and mode on `insert()`/`send()`, and the facade),
`scripts/model_chain.py` (settings and route registration), `style.css`.

---

## 2. The three decisions worth knowing before reading the code

**A revision is a comparison, and the lock only makes the comparison mean
something.** Nothing here serialises writes for the sake of it. The critical
section is a file read, a dictionary copy and a file write; a model is never
loaded inside one, a socket is never written inside one, and nothing waits for a
card inside one.

**A file written before revisions existed compares by the SHA-256 of its bytes.**
That is what lets reading stay read-only. A store that wrote a revision on read
would rewrite every file in the folder the first time somebody opened the thread
list, and "reading never writes" is worth more than a tidier data model.

**A reply belongs to the server, and both views subscribe.** The Gradio
generator is a follower that writes nothing, so closing it — by refreshing, by
`cancels=`, by navigating away — loses a subscription and nothing else. This is
not a new pattern in this repository; `mc_llm_jobs.py` has run MiniMax's
requests this way since it was written.

---

## 3. Where the specification was wrong, and what was done instead

### 3.1 The "orphaned" spatial popup CSS (spec §15.1)

The audit lists `.mc-krea-spatial-menu` and `.mc-krea-spatial-popup` as real
popup CSS with **zero Python or JavaScript references**, left behind by the
reverted Klein region geometry, and says to delete them separately.

They are not orphaned. `model_chain_krea_creative.spatial_editor()` emits all
three classes today; the markup builds the names in pieces, which is why a
literal search for them finds nothing and why the audit reached the wrong
conclusion. They were deleted, the existing spatial stylesheet test caught it,
and they were put back. `tests/test_overlays.py` now records the finding, so a
second reader following the audit does not delete them again.

### 3.2 The completion guard needed a second comparison (spec §7.5)

The specification says the completion transaction runs with
`expected = accepted_revision`. That is necessary and not sufficient: a revision
only moves when this code writes, so a file changed by something outside the
application — an editor, a sync client, a second process — comes back with
different bytes and the same number, and the reply would be appended to a
conversation that is no longer the one it answers.

So an operation also captures the conversation's exact bytes at acceptance and
the completion compares those too. One extra file read, once per operation,
between accepting the command and starting a model load measured in seconds.
`transaction()` grew a `guard` parameter for it.

This was found by mutation-checking: removing the revision comparison left every
test passing, because the test that named it was being caught by the target
index instead.

### 3.3 Receipts are per step, not per operation (spec §5.6)

An operation can write twice — the acceptance (a user turn, or a branch) and the
completion — and both land in a file that remembers operation ids. Sharing one
id makes the second write look like a retry of the first, and the transaction
answers "already applied" and writes nothing. A branching regenerate lost its
reply to its own operation's earlier step, silently. The completion is written
under `<operation_id>#reply`.

### 3.4 The speech turn is created at acceptance, not in the executor (spec §7.4)

The specification puts `begin_speech` inside the execution loop. The browser has
to have the turn id in the *first* frame it is sent, or it cannot open the audio
stream while the model is still thinking — and a frame carrying an empty id
reads as Voice being switched off. It is created on the acceptance path, before
the executor starts.

---

## 3.5 Four defects found in use, and what they were

Reported against the first build, on a running Forge. Two more followed against
the second build; those are §3.6.

### The panel could not be minimised, and the workspace menu would not close

One missing CSS rule, and it presented as three separate bugs.

`hidden` is how every piece of this panel is shown and hidden from JavaScript,
and the browser implements it as `[hidden] { display: none }` in its
*user-agent* stylesheet. **Any author declaration of `display` beats a
user-agent one outright, whatever the specificity** — so
`.forge-assistant-panel { display: flex }` made the panel unhideable, and the
same for `.forge-assistant-menu` and the conversation body. ✕ set an attribute
and changed nothing anybody could see.

Fixed with one rule scoped to the assistant's root, carrying `!important`
because what it overrides is the `display: flex` rules a few lines below it and
a plain declaration would tie and lose on source order. `tests/
test_assistant_js.py` asserts the rule exists and that everything hidden from
script is inside the root it is scoped to.

### The workspace menu also waited for a confirmation that never came

Independently of the above: the menu closed in the `.then()` of
`activateWorkspace`, whose watchdog rejects after four seconds if
`getActiveWorkspace()` never reports the destination. On a page whose tab bar
the adapter reads differently from the way it expected to, that is never. The
menu now closes on the press; the highlight still follows the host's own
selection, and a failed switch is reported in the status line.

### The utility menu was assembled from whatever was in the header

It walked `#quicksettings` and the footer and offered what it found, which on a
real installation is "Apply settings", "Reload UI" and a column of controls
whose only visible text is the word JSON. Replaced with two written-down
entries — Unload All Models and Unload LLM — behind a route
(`POST …/v2/unload`) rather than by pressing a control in the page, because the
Settings page's Actions row is a button that may or may not exist under a given
theme.

### Focus mode covered nothing and stopped the page scrolling

Two causes, both of them this implementation doing more than the reference one.

`Host.panels()` paired tab buttons with `#tabs`'s children that carry an id.
One of those candidates *contained the tab bar*, so the class went onto a box
holding the tab bar and the workspace together — laid over the window, tab bar
and all. `panels()` now excludes any candidate containing `.tab-nav` or a
`role="tablist"`, with a test.

And `enter()` locked the body's scrolling and walked up the tree setting
`inert` on every sibling at every level. Neither is in the reference
implementation and neither is needed — the focus root covers the viewport, so
there is nothing behind it to reach — and between them they produced the visible
half of the report. Focus is now what the reference document describes: add a
class, and that is the whole of the DOM work. `overscroll-behavior: contain` in
the stylesheet keeps a wheel gesture inside the workspace without touching the
page at all.

The pin was removed at the same time. It was a preference for "keep the panel
open across workspace changes" occupying the corner everybody aims at for
"close this"; the panel now simply stays open.

## 3.6 Two more, reported against the second build

### The panel never showed a conversation

It came up on "Reconnecting… Your draft is safe." with an empty transcript and
stayed there. Nothing was wrong with the transport.

The panel never learned **which** conversation to show. `Store.select()` sets
the selection and is what every render reads; nothing called it. The selection
stayed empty, `refresh()` returned early because there was nothing to ask
about, and the transcript was never going to be given anything to draw. The
missing piece was not a bug in a function — it was a function with no caller,
which is the kind of gap a unit test passes straight over.

Three parts to the fix, and they are three because a selection has three
moments:

*Where a new page starts.* `bootstrap` now answers `selection` — the character
and thread LLM Studio was last left on, read from the same preference the tab
uses — along with `characters` and `mode`. A **seed and not a source of
truth**: the preference is installation-wide, so a second window deliberately
put on another thread keeps it. The store applies the seed only when its own
selection is empty.

*Where it goes next.* `_selected()` in `mc_llm_chat_panel` is called by all five
ways of landing on a thread (choosing a character, opening a thread, a new
thread, branching, following a generation) and does two things with one fact:
remembers it, and publishes `character_changed`. The store's `follow()` moves
the panel to match. Without it the panel seeded itself once and then never
moved again, which is two views on different conversations with nothing on
screen to say so.

*What it can switch to on its own.* `snapshot` now carries `threads` for the
character and `characters` for the installation, which is what the panel's own
picker lists.

Publishing fails open here as everywhere: a window that cannot be told is one
selection behind, which is a refresh away, and a selection that *failed* would
be a tab that would not change threads.

The status line was also lying. "Reconnecting" is only true the second time,
and it was the first thing on screen on a feed that had never opened — a
sentence pointing at the wrong problem. It says "Connecting…" until a feed has
opened once, and "Pick a conversation to begin." when there is a connection and
nothing chosen, because an empty transcript under the word "Ready" reads like a
conversation that lost its messages rather than one that was never picked.

### Focus mode still left the theme's header on screen

The first fix made focus mode what the reference document describes: one class,
one rule, `position: fixed; inset: 0` and a z-index. That is correct, and it is
not enough.

**A z-index wins over ordinary content. It does not win over a header that is
positioned and has a stacking context of its own** — which is exactly what a
theme that draws its own chrome produces, and Lobe draws its own chrome. The
workspace was nominally on top of the header and the header stayed exactly
where it was.

Covering is therefore the wrong mechanism. The chrome is now taken **out of the
layout**: on entering, the workspace's ancestors from its parent up to `<body>`
are marked `forge-assistant-focus-path`, and one stylesheet rule hides every
child of a marked element that is not itself on the path, not the workspace,
and not the assistant. That is the tab bar, the theme's header and sidebars,
the footer and the other tabs — without this code having to know what any of
them are called, which is the point: a theme it has never heard of is hidden by
the same rule.

Two details worth keeping:

*Dialogs are excluded* (`[role="dialog"]`, `[aria-modal="true"]`), because a
modal the host opened over the page is not chrome.

*`[hidden]` is deliberately not excluded.* Something already hidden stays
hidden and comes back hidden.

A second rule hides `.tab-nav` and `[role="tablist"]` anywhere on the page
while focus is on. The path rule already finds the tab bar where Forge leaves
it; a theme is free to move it, and a rule keyed on what the bar *is* rather
than where it sits costs nothing.

Leaving is the classes coming off. No inline styles to restore, nothing moved
in the DOM, and a workspace rebuilt by a Gradio update while focused is still
handled, because the exit path removes the class from the saved node whether or
not it is still attached.

One related change: a containing-block trap (a `transform`, `filter`,
`perspective`, `contain` or `backdrop-filter` on an ancestor, all of which make
`position: fixed` resolve against that ancestor rather than the viewport) used
to **refuse** focus. It is a **note** now. Taking the chrome out of the layout
means a workspace that cannot quite escape its ancestor still fills what is
left of the window, which is the thing somebody asked for; refusing outright
meant a press that did nothing at all and a sentence nobody could act on. The
note is shown in the status line and names the property it found.

---

## 4. Deliberate deviations

**No `DataTransfer` adapter for the tab's image input.** The specification names
it as a candidate for keeping the tab's visible chip in step with a pasted
image, pending gate G7. It is a guess about what the installed Gradio does with
a synthesised file drop. The paste bridge hands Python a *staging token*
instead: the browser uploads, the server validates and decodes, and a hidden
callback puts the resulting picture into the chip. That needs no unverified host
behaviour and the chip shows exactly what will be sent.

**No write-ahead journal.** Tier 2 (spec §5.7) is specified and deferred, as the
specification allows. Single-file writes are atomic; a crash between the two
files of a branch can leave an empty file, never a corrupt one, and
`mc_llm_conversation_startup` names any it finds in the log rather than deleting
them — an empty file is somebody's conversation as far as that code can tell.

**Message editing in the flyout uses the browser's own prompt.** A full in-panel
edit buffer with Save/Cancel is specified (§13.4) and is the obvious next piece
of work; what is there now is correct — it carries the same (index, version,
revision) and is refused the same way — and visibly plainer than the tab's.

**`mc_llm_state.remember()` is left installation-wide.** As §23 says it should
be. It seeds a page's initial selection once; live selection is page-local, so
two windows can deliberately be on different threads.

---

## 5. Gates: what could and could not be resolved here

Forge itself is not in this checkout, so the gates that need it are recorded
rather than closed.

| Gate | State |
|---|---|
| G1 Forge/Gradio versions and event signatures | **open.** The code uses only `click`, `submit`, `change`, `then`, `success` and generator outputs, all of which this repository already relies on. `cancels=` was removed from Stop; if it must stay for a host reason it is harmless, because the operation is not in the generator. |
| G2 Theme mount points, portal roots, listener order | **closed for the case that was failing, open in general.** Focus no longer covers the chrome, it takes it out of the layout: the workspace's ancestors are marked and one rule hides everything else under them, so a theme's header is hidden whatever it is called and wherever it sits (§3.6). That is what a z-index could not do under Lobe. A containing-block trap on an ancestor is a note rather than a refusal now, because the workspace still fills what is left of the window. Listener order remains unverified against a running host. |
| G3 Installed tab ids and header controls | **open.** `forge_assistant_host.js` reads the host's own `#tabs` element, pairs buttons with panels by position, and offers whatever it finds. It never matches a tab by display name. |
| G4 Whether `script.js` honours `defaultPrevented` | **open.** No document-wide Send or Stop shortcut is installed, so nothing depends on the answer. Escape uses a precedence function with its own tests; the one thing it will not do is swallow Escape when nothing closer to hand wants it. |
| G5 Hidden field value before its tab is selected | **open.** The capability is rendered into `mc-llm-chat-conversation-key` exactly as `mc-llm-chat-voice-key` is, which is the pattern already proven in this installation. If it turns out to be empty before the tab is selected, the fallback is a bootstrap route behind the host's own auth; the store already treats a missing key as "conversation unavailable" and keeps navigation and focus working. |
| G6 Whether generation can run outside a Gradio worker | **closed enough to ship, in this checkout.** `mc_llm_sessions.conversation()` takes a request and a cancellation and yields events; the tests drive it from a plain thread. If a host turns out to need request context, the executor is the one place an adapter goes. |
| G7 Gradio image input via DataTransfer | **not needed.** See §4. |
| G8 `gr.Chatbot` row classes | **open, and the one thing here that can fail silently.** Three candidate selectors are used; a Gradio or theme upgrade can stop all three matching, which costs bubble width and nothing else. Add a visual check to the upgrade checklist. |
| G9 Base path and secure context | **partly closed.** The base path is read from the document's own URL and the routes are registered under it, with a test. Secure context is a deployment fact: without HTTPS there is no microphone, and `voice_chat.js` already degrades cleanly. Image paste works either way. |
| G10 Filesystem atomic replace and fsync | **closed for the write.** `atomic_write_json` fsyncs the file and renames; `mc_llm_conversation_store.fsync_directory()` is there for the rename, is never fatal, and is a no-op on hosts that do not support it. |
| G11 Test environment | **closed.** `pip install pytest pillow numpy httpx` and the suite runs. Baseline at `78da206` was 6,113 passing, 13 skipped, zero failures; this work leaves it at 6,405 passing, 13 skipped. |
| G12 Traces of send/regenerate/Stop/refresh under contention | **open.** Needs a GPU and a running host. |

---

## 6. Acceptance matrix: what is covered by tests here

Covered, with a named test: **P01–P04**, **H01, H03–H08, H10–H14**, **T01–T03**,
**O01–O02**, **C01**, **X02** (path traversal, oversized and decompression-bomb
images, malicious Markdown, mismatched attachment ownership), and the store,
service, operations and feed halves of **V05**.

Partly covered: **H02** (one adoption commits; the two-view race needs two
processes), **H09** (the Stop race is covered; "no speech after an accepted
Stop" is asserted from the completion record rather than from an audio device),
**T04–T07** (the reducer is covered; the transport fallback needs a network),
**G02** (tie-breaking, threshold and cancel are covered; the pointer devices are
not).

Not covered, and needing a browser or a host: **P05–P06**, **G01, G03–G10**,
**C02–C08**, **O03**, **V01–V04, V06–V07**, **X01** (the gates are covered; a
real unauthenticated request is not), **X03–X04**, **X06**.

Nothing untested is claimed as passing.

---

## 7. Adding things

**A workspace focus adapter.** `forgeAssistant.focus.registerAdapter({workspaceId,
canFocus, enter, exit, onResize})`. `canFocus` returns `{ok, reason}` and the
reason is shown to the person. The rule that matters: never change a canvas's
bitmap dimensions — that clears it. Call the component's own fit method, and if
there is not one, do nothing. A picture that has to be scrolled is a far smaller
problem than a picture that has been erased.

**A message action.** Add it to `ACTIONS` in `mc_llm_conversation_service.py` and
to `_apply()` if it is a plain content change, or to the `GENERATING` tuple and
`_plan()` if it ends in a model being asked for words. It is then available from
both views with identical semantics, because both views send the same envelope
through the same door. Renaming anything already in `ACTIONS` is a breaking
change; adding to it is not.

**A pop surface in LLM Studio.** Add it to `OVERLAYS` in `mc_llm_overlays.py` and
register its component during the build. Everything else — one at a time, in
both directions, across all forty-two ordered pairs — follows.

---

## 8. Migration and rollback

Revisions are lazy. Nothing is rewritten in bulk, no file is touched until
something writes to it, and a chat this extension has never written still looks
exactly as it did.

Downgrade is **not write-interoperable**. An older copy of the extension reads a
v5 file perfectly well — `Conversation.from_dict()` has always tolerated unknown
keys — and drops the revision and the receipts the next time it saves one. That
is not corruption and no message is lost; it is the guarding going away for that
file until something writes it again. Back the chats folder up before testing a
downgrade, and test it on a copy.

To roll back: check out the previous commit. There is no data migration to
undo, no schema to revert, and no setting that has to be turned off first.
