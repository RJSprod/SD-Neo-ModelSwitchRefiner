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
the second build (§3.6), two more against the third (§3.7), a fifth report
finally produced the real diagnosis of all four focus-mode reports (§3.8), and
one more came in once focus mode worked (§3.9). §3.10 is the tab's own
transcript, which is a separate implementation and had a separate defect, and
§3.11 is three asks against the panel once the whole of it worked, and §3.12
three more against its own chrome, and §3.13 is a cross-extension one. §3.19 is
the second round of asks: read aloud, tap to reveal, Send to prompt, Send again
and Auto Attach.

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

> **This diagnosis was wrong, and the fix made things worse.** See §3.8. The
> candidate that "contained the tab bar" was Txt2Img itself, whose extra
> networks are a nested `gr.Tabs` on every Forge there is. Excluding panels
> for containing a tab bar excluded most of them, Txt2Img first.

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

A tab bar the theme has moved somewhere the path rule cannot reach is marked
too — but **by the script, one element at a time, never by a selector**, and
that distinction is §3.7.

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

## 3.7 A third round, and a rule that cost the page

### The blank page

The first attempt at §3.6 shipped with a second stylesheet rule beside the path
one: `.tab-nav` and `[role="tablist"]` hidden anywhere on the page while focus
was on. It was added as a free safety net for a theme that relocates the tab
bar, it reads as an improvement in a diff, and on a real host it hid
everything.

Two reasons, either of which is enough:

*Those names are on every nested tab group.* A `gr.Tabs` inside a workspace
carries the same class and the same ARIA role as the one at the top of the
page. Txt2Img is full of them. The rule reached inside the workspace it was
meant to be filling.

*The role need not be on the strip of buttons.* On a host where `role="tablist"`
sits on a container rather than on the buttons, that container is an **ancestor**
of the focused workspace, and hiding an ancestor hides the workspace with it.

The condition that makes hiding a tab bar safe is "unless it contains the
workspace", and there is no way to write that as a selector. So `enter()`
decides: it walks the tab bars, skips any that is the root, is inside the root,
or contains the root, and marks the rest with a class whose only rule is
`display: none`. The stylesheet does as it is told and decides nothing.

`tests/test_assistant_focus_js.py` asserts both halves — that a relocated bar is
marked, that a nested one and an ancestor one are not — and asserts that **no
selector in `style.css` names a tab bar at all**. That last one is the test that
would have caught this.

### The safety valve

The deeper lesson is that the rule which takes the chrome out of the layout is
written against a shape this code cannot see: somebody else's theme, on somebody
else's Forge. Getting it wrong does not cost a misaligned panel, it costs a page
with nothing on it and nothing to say why — and the code has no way to know it
happened.

So it asks. After the marking and before anybody looks at the result, `enter()`
measures the workspace. If it is not being painted, every mark this module made
comes off and what is left is the plain overlay: the reference implementation's
behaviour, imperfect under a theme that draws its own header, and never blank.
The mode stays on, so Escape and the Focus toggle still leave it, and the status
line says what happened.

A host with no layout to ask is given the benefit of the doubt. Throwing away a
mode that may well be working, on a question that could not be answered, is the
wrong default.

### The transcript did not open at the latest message

Reported alongside: the thread showed, but not at the end of itself.

The follow-the-bottom machinery was all there — `following`, the slack, the jump
button — and three things were missing around it.

*`following` and `lastRendered` were per panel, not per conversation.* A thread
left scrolled halfway up put the next thread halfway up too, at a pixel offset
measured against a message that was no longer on the page. The selection epoch
is now compared on every render, and arriving at another conversation resets
both.

*Scrolling to the end happened once, synchronously.* That covers the ordinary
case, because replacing the messages forces the layout that answers "how tall is
this". It does not cover a box that arrives later — a panel opening, a
conversation section expanding, a picture loading in a bubble — and a transcript
that scrolled to the end of nothing is a transcript at the top. It is done again
on the next frame, and again when anything inside the transcript fires `load`,
in both cases only while still following.

*A redraw skipped for an unchanged fingerprint skipped the scroll with it.* The
content can be identical while the box it is in is not, which is exactly the
case above. The early return now re-pins before it returns.

Two defects fell out of writing the tests for this, neither reachable from the
outside:

**Bubbles were reused across a revision.** `updateBubble` refreshes a reused
node's text and its provisional class. It does not rebuild the action bar, whose
buttons closed over the row and the revision they were *built* with and send
those. A bubble reused after the thread moved sends a stale revision on every
action, and every one is refused. The bubble key now carries the thread and the
revision; a reply arriving token by token does not move the revision, so the
per-token redraw the keying exists to avoid is still avoided.

**A conversation that went away and came back stayed away.** A snapshot can
arrive without its conversation. The empty branch cleared the transcript and
left `lastRendered` pointing at what had been in it, so when the conversation
returned unchanged the redraw was skipped and the transcript stayed empty.

## 3.8 What was actually wrong with focus mode

Four reports in a row said focus mode did not work, and each fix answered the
report it had rather than the cause — the body-scroll lock, then a cover that
could not beat a sticky header, then chrome taken out of the layout, then a
safety valve. Every one of them was built on the assumption that the host
adapter was handing focus mode the right element. It was not, and there was a
second, independent reason nothing was ever visible. Both were found by reading
`sd-webui-lobe-theme` and Gradio 4.40's `Tabs.svelte` rather than guessing at
the DOM a fifth time.

### The host adapter paired the wrong things

Gradio renders `gr.Tabs` as

```
div#tabs.tabs
  div.tab-nav[role=tablist]
    button[role=tab][aria-controls=tab_x][id=tab_x-button] …
  div#tab_x.tabitem[role=tabpanel][style="display: block|none"]
  …
```

and renders every **nested** `gr.Tabs` — Forge's extra networks inside Txt2Img,
the mode tabs inside Img2Img, the pages inside Settings — with exactly the same
classes and roles. `forge_assistant_host.js` ignored that:

- `bar()` collected every button under `#tabs` whose parent was a `.tab-nav`,
  which is every nested tab button on the page as well;
- `panels()` (from §3.5) threw away every panel with a nested tab group inside
  it, which is Txt2Img and most of the others;
- `listWorkspaces()` paired those two corrupted lists **by position**, and
  `getActiveWorkspace()` walked the filtered panels for a visible one.

On Txt2Img the "active workspace" therefore resolved to a hidden panel, or to
nothing. Focus mode then did exactly what it was told: marked that hidden
panel's ancestors and hid everything else — including the real workspace —
which is the blank page of §3.7. With the valve in place it degraded instead,
and the header stayed. The picker was mispaired the same way.

Now: the top-level bar is the **shallowest** `.tab-nav`/`[role=tablist]` under
`#tabs` (a theme may wrap it; a nested one is always inside a panel and so
deeper); panels are `#tabs > [id^="tab_"]`, excluding only a candidate that
*contains the bar*; buttons are paired with panels by `aria-controls`, which
Gradio writes on every tab button, then by the `<id>-button` name, and only
then by position. `visible()` reads the computed position, because the focused
panel is fixed by a class and its inline style says nothing.

### The focus rule never applied

Gradio styles a tab panel from a Svelte component whose rules are scoped with
a generated class: what ships is `div.svelte-<hash> { position: relative; … }`.
A tag plus a class outranks a lone class, so `.forge-assistant-focus-root {
position: fixed }` **lost, silently, on every Forge**. The panel stayed a row
in the page. This is why the very first report — "the menu bar does not go
away, the page stops scrolling" — showed the scroll lock and nothing else: the
lock worked, the overlay never did. The rule carries `!important` now, and a
test reads the stylesheet to keep it that way.

### What Lobe actually does

Read from its source (`src/main.tsx`, `src/app/index.tsx`, `useInject`,
`useNavBar`): it appends `<div id="root">` to `<gradio-app>`, renders its
header (`position: sticky; z-index: 999`), a `<main>` with sidebars, and a
footer inside it, **moves Gradio's `.app` container into that `<main>`**, hides
`#tabs > .tab-nav` with an inline style, and draws its own tab strip in the
header — a tablist that is not under `#tabs` and whose entries click the hidden
Gradio buttons. Nothing in that needs special handling once the adapter reads
the page correctly: the header, sidebars and footer are unmarked children of a
marked ancestor and the path rule hides them; the theme's own tablist is
outside the workspace and `enter()` marks it. `tests/test_assistant_js.py` now
builds this exact shape (and Forge's plain one, nested tabs included) and runs
the adapter against it; `tests/test_assistant_focus_js.py` enters focus on it.

`enter()` also writes one line to the console saying which panel it focused,
how many ancestors it marked, how many bars it hid and whether it degraded.
That line would have shortened this by three rounds.

## 3.9 The gap above the gallery

Reported once focus mode finally worked: a blank band above the Generate
button and the gallery, with both sitting at the same place on the screen
whether focus was on or off, while the prompt column moved up to fill the
space.

The header was gone; its *reservation* was not. Lobe's split previewer (from
`src/styles/components/container.ts`) makes the results column
`position: sticky; top: 80px !important` — 64px of header and a margin — so the
picture stays on screen while the prompt column scrolls under it, and it moves
the Generate button into that column. In focus mode the workspace is its own
scroll container and there is no header inside it, so the same 80px is nothing
but a gap. Because a sticky offset is measured from the top of the scroll
container either way, the column lands at the same screen position with the
header hidden as with it shown — which is the observation in the report.

`enter()` now reasons about this generally rather than naming the theme's ids:
a sticky element whose offset is measured against the workspace's own scroll
edge — no scroller of its own between it and the root — was clearing something
above the workspace, and everything above the workspace is hidden. Its offset
becomes the root's padding, so a stuck column keeps the margin it has at rest.
A sticky element inside an inner scroller is measured against that scroller
and is left alone, and so is one whose offset is no larger than the padding.

This is the one thing focus mode does with an **inline style**, and the reason
is specificity again: the theme's declaration carries `!important` on an id,
and nothing in a stylesheet outranks that reliably. The value it replaces —
and its priority — is recorded exactly and put back on the way out, by the
safety valve as well as by `exit()`. Hidden subtrees are skipped whole during
the walk and there is a node budget, so a toggle stays a toggle on a page with
thousands of elements in inactive nested tabs.

## 3.10 The tab's bubbles: three symptoms, one wrong element

Reported in use, against the Conversation tab rather than the panel: user
messages that did not line up with each other, replies whose cell ran the full
width, and a two-word message — "hey there" — wrapped onto two lines.

All three were one mistake, and it is invisible in the stylesheet unless you
know what `gr.Chatbot` renders. Gradio 4.40's is

```
div.message-row.bubble.user-row      the flex row, which places the bubble
  div.avatar-container
  div.flex-wrap.user                 THE BUBBLE: border, radius, padding
    div.message.user                 the text box
      button > the markdown
```

with `.flex-wrap { width: 100% }`, `.message { width: calc(100% - xxl) }`, a
`max-width` on the **bot** row only, and none at all on the user row.

What was here set `max-width: 75%` on **`.message`** — the text box inside the
bubble, not the bubble:

- the bubble was never constrained, so a reply's cell ran to the row's own
  limit, which is "the full width" in the report;
- the text box was squeezed to 75% of a bubble that had already been sized for
  100% of the text, so the text *had* to wrap — at three quarters of the space
  it was given. That is the two-line "hey there";
- the leftover quarter was visible, because `.message.user` carries the accent
  stripe: it was drawing 25% short of the bubble's trailing edge rather than on
  it. The stray vertical bars inside the bubbles in the report are that stripe,
  and they are what confirmed the diagnosis before a line was changed;
- and because Gradio shrink-wraps the user row to its content while its child
  asks for a percentage of it, the row's width was content-dependent. No two
  user messages agreed on where their right edge was.

The refactor states the geometry against the real shape: the row is stretched
and justifies its one bubble to its own side; the bubble hugs its text
(`width: fit-content`) up to its share of that row; the text box fills the
bubble (`width: auto; max-width: none`), which is the declaration that makes a
line wrap when the bubble is full and not before. `min-width: 0` on the bubble
so one unbreakable token cannot push the thread sideways.

Nothing touches the margins, the radius or the padding. With avatars on,
Gradio's own gutters are already symmetric — two spacing units outside each
bubble on the side it sits against — so the outer edges line up without this
file having an opinion about them, and those remain the theme's business. No
`!important` anywhere: `#mc-llm-studio .mc-llm-transcript …` outranks every
rule it needs to beat.

Both halves are keyed on `.flex-wrap` deliberately. It is Gradio's class rather
than ours and an upgrade can rename it; keyed on the same class the two rules
stop matching **together**, and what is left is the component's own layout
rather than a bubble this file has constrained around a text box it can no
longer reach. That is what closes G8, which had this down as a silent-failure
risk while the failure was already happening.

One process note. These rules had escaped the convention the repository already
had — `test_every_rule_that_names_a_gradio_class_names_no_generated_one`
requires a selector reaching into Gradio's classes to be scoped
`#mc-llm-studio .mc-llm-…` — because they were written against
`#mc-llm-chat-transcript` instead, which that test does not look at. They are
in scope now, and `flex-wrap` was added to the classes it checks.

## 3.11 Three asks against the panel once it worked

### A menu you could not leave

Both header menus had no way out but to choose from them. Escape closes them,
and so does opening the other, and so would a second press on the button that
opened it — none of which is any use to a thumb: a phone has no Escape key, and
those two buttons are a small target beside a menu that is covering them.

Every menu ends with **Cancel** now, appended in `toggleMenu()` rather than in
each builder, so a menu added later cannot forget one. The threads menu got it
too, though only the other two were reported: it is the same menu with the same
problem.

### The picker would not switch while focused

Focus hides every panel but the one it is filling, and it hides them with
`display: none !important` — which beats the inline `display: block` Gradio
writes on the panel it has just switched to. So with focus on, pressing a tab
button changed the host's selection and changed nothing on screen: the
destination stayed hidden, the old workspace stayed fixed to the viewport, and
`getActiveWorkspace()` — which answers with the panel that is *showing* — went
on naming the focused one. The switch could not even be **observed**, so
`activateWorkspace`'s confirmation watchdog ran its four seconds out and
reported that the workspace had not opened.

`switchWorkspace()` takes focus off, asks the host, and puts focus on at the
destination. Off first, because the host cannot show a panel this code is
hiding. Back on afterwards, because somebody in focus mode who asks for another
workspace is asking for that workspace, not for the end of focus mode. And back
on at the *old* one if the switch fails, because a switch that did not happen
should not cost the mode either.

One interaction this introduced and closed in the same change: between the exit
and the re-entry, focus is off, and the navigation subscriber's `moveTo` refuses
with "Focus is not on." — which would have turned a working switch into a
warning and a lost mode. The subscriber now only moves focus that is on, which
is the correct condition on its own terms.

`toggleFocus` and the picker share `refocus()`. Entering focus and moving it are
one operation with the same three outcomes — refused, entered with a caveat,
entered cleanly — and only one of the two callers used to report all three.

### Eight buttons and a pager under every bubble

At panel width that wrapped onto three rows under each message, so a thread was
more chrome than conversation; and the actions aimed at a message in the
*middle* of one were the heavy ones — branch, truncate, renumber — which want
the room the tab has to explain themselves in.

The panel offers three, as icons, on the newest message only: edit, regenerate
(replies only — an unanswered message of yours has nothing to ask again), and
delete. `\u21bb` is the glyph the tab already draws for regenerate, so the two
views agree on what it means. Each carries `title` and `aria-label`, because an
icon with no accessible name is a button only sighted people have, and that is
exactly what goes wrong when labels come off.

`act()` lost the branches nothing reaches any more — the Listen dispatch, the
truncation confirm, the version-pager target rewriting. Dead code that looks
live is how the next person concludes this view still does all of it.

**This is a deliberate narrowing of specification §11's "every message
action".** It is recorded in §4 with the rest of them. Nothing was removed from
the conversation: the service still implements all sixteen actions, the tab
still offers them, and the guarding is unchanged. What changed is one view's
opinion about which of them belong on a phone-width panel.

## 3.12 Three more, against the panel's own chrome

### A menu the panel cut off

With the conversation collapsed, the workspace list was clipped and its last
item — Cancel, added in §3.11 — could not be reached.

The menu was an absolutely positioned child of the panel with
`max-height: 50vh`, and the panel clips what it contains (`overflow: hidden`,
which is what keeps the conversation inside the rounded corners). A collapsed
panel is a header and an accordion tall, so most of a 50vh menu was outside
that box. Worse than merely hidden: **unreachable**, because scrolling a menu
whose visible region is shorter than its own scroll viewport cannot bring the
bottom of it into view. The control added to let people out of a menu was the
first thing cut off it.

`placeMenu()` fixes the menu to the window instead, sized to the room that is
actually there and opened upwards when there is more room above — which there
is whenever the panel is docked along the bottom, where a downward menu has
only the few pixels between the header and the edge of the screen. A floor of
120px and `overflow-y: auto` so a window with no room either way still gets a
menu it can scroll rather than no menu at all.

`top` in both directions and never `bottom`: `bottom` on a fixed element is
resolved against the layout viewport while everything else in this file is
measured against the visual one, and mixing the two is how a panel ends up
behind a phone's keyboard. Opening upwards therefore costs one measurement of
the menu's own height, taken after the cap is applied so it is the height the
menu will really be drawn at.

The panel keeps its clipping. Nothing about the conversation changed.

### Free Float

The six anchors exist so that a window resized between sessions cannot leave
the panel off-screen: what is stored is *which corner*, and a corner is a
corner at any size. Free float has to store a position, so it stores **how far
across the available travel** the panel was — 0 against one edge, 1 against the
other. That maps onto any later viewport and can strand the panel no more
than an anchor can, which is the property the anchors were protecting and the
one thing a remembered pixel would have lost. `floatPoint()` clamps inside the
same safe-area insets and gap as `anchorPoint()`.

Turning it on adopts the panel's current position rather than jumping: a mode
that moves the thing you were looking at is a mode people turn off again to
find it. Dragging in free float previews nothing and chooses no anchor, because
there is nothing to snap to.

It is the one piece of layout state kept in `localStorage` rather than
`sessionStorage`, under a key of its own. A panel left open, a width dragged
wider and a corner chosen are this tab's business; whether the panel snaps to
corners **at all** is a preference answered once, and the user asked for it to
be remembered. Keeping it out of the session payload also means it survives
that payload being rejected by a future schema bump.

It does not apply to the phone sheet, which is anchored to a half of the screen
and covers it. The preference stays on and applies again on a wider window,
which is tested rather than assumed.

The entry is a `menuitemcheckbox` with `aria-checked`, so it reports its state
instead of only acting, and it is added in `utilityItems()` rather than in the
host's `listUtilities()` — that list is about things the *host* can be asked to
do, and it has a test saying so.

### One header row

Two rows of chrome — a title row with the panel's name and the ✕, a nav row
with Workspace, Focus and ⋯ — above a collapsed conversation was most of the
panel. The name was the one thing on screen nobody needed telling, there being
exactly one floating panel on the page; the panel keeps it as its `aria-label`,
which is what a screen reader announces on entry and the only place a name was
doing any work.

One consequence needed handling rather than discovering: `startDrag` ignores a
press that lands on a control, so a single row of nothing but controls would
have left nothing to drag the panel by. `.forge-assistant-grip` is that space,
explicitly — `flex: 1 1 auto`, `aria-hidden`, no appearance of its own — and it
is also what right-aligns the ✕. The header wraps rather than overflowing if a
theme's font makes the four controls wider than the panel.

The dead `.forge-assistant-title` and `.forge-assistant-nav` rules went with
them, with a test asserting their absence: CSS naming a class nothing emits is
the next person's wrong picture of the panel.

## 3.13 Getting out of another extension's way

Reported from a page running Mini Paint NEO beside this one: its Send to WanGP
popup could not be used with the assistant on screen.

Two causes, and neither was in this file. The popup draws at `z-index: 60`,
which is above everything that extension draws and below everything this one
does — 1100 for a focused workspace, 1200 for the panel — so in focus mode it
was covered outright and out of it the launcher sat on top. (It was also under
Mini Paint's *own* canvas focus mode at 1000, and had been since both were
written; the report came from here because this is what the user had open.)
That half is fixed where the number belongs: `--minipaint-dialog-layer: 2000`
in that repository, with the popup and its toast drawn on it.

The half only this side can do is the panel. Stacking decides what covers what;
it does not move 360 pixels of conversation off a dialog somebody is reading,
and no number can. So Mini Paint publishes an event when its popup takes the
page and when it gives it back —

```js
document.dispatchEvent(new CustomEvent("minipaint:overlay", {
    detail: {name: "intercept", open: true, modal: true}
}));
```

— and `Shell.prototype.yieldTo` puts the panel back to its launcher. Three
decisions in it are worth keeping:

*Only `open` acts.* The event fires both ways and only one direction is a
request for room. Acting on the other closes the panel of somebody who has just
dismissed a popup and gone back to work. The guard for this was masked at first
by the already-closed guard, and it took a test with the panel open and a
closing event to pin it.

*Nothing reopens.* A panel that springs back over the page somebody has just
returned to is the same complaint from the other end.

*Focus mode is untouched.* The dialog draws above it now, so there is nothing to
leave, and a workspace that emptied itself whenever a popup opened would be
worse than the bug.

Neither side imports the other or knows the other's class names. The whole
contract is one event name and one number — and this repository carries a test
asserting its own two layers stay below that number, because if either ever
climbed past it the popup would go back under the page and the symptom would
look like anything but a z-index.

---

## 3.14 The collapsed panel had nothing in it

> "I like how the flyout looks when expanded, but when its collapsed, its
> useless. Lets make is useful in this view. When conversation is collapsed, i
> should see all the options of workspace side by side as a button row, only
> bounded by the width of the screen. After i select, the flyout to turn back
> into a button (as if i closed it)."

Accurate. Collapsed, the panel was a header and a folded accordion: four
controls, none of which did anything without opening something else first. It
was strictly worse than the launcher it came from, which at least took one
press to get somewhere.

The one thing a small always-on-top panel is well placed to be is a tab bar, so
that is what it is now. `renderWorkspaces` draws every entry `listWorkspaces`
returns into a row under the header, and `applyAccordion` shows that row
exactly when the conversation is hidden.

**Rebuilt, not patched.** It is a handful of buttons redrawn on two events —
the accordion closing, and the host's selection moving — and a diff of that
would be more code than it saves.

**The press closes first and switches second.** Either order switches the tab;
this one does not leave the page changing underneath a panel that is on its way
out. A switch that fails still reports, in the status line of the panel that
comes back.

**The mark is the host's, not the press's.** `aria-current` is set from
`getActiveWorkspace()` on every draw, and the navigation subscriber redraws.
Set from the press instead, it would sit on whichever tab was last pressed here
and disagree with the page as soon as anybody used a tab button — the same
mistake the picker was written to avoid, and the reason the navigation
subscriber has its own test.

**The width was the row's, for one round.** `placeNow` wrote `panelWidth` — 360
by default — whenever the panel was open, and a tab bar under a 360-pixel cap
is a tab bar in four lines, so collapsed it wrote no width at all and the
stylesheet took over with `width: max-content` bounded by the viewport.

That is not what it should have been, and the screenshot that came back said so
plainly: eleven tabs, a panel most of the way across a 1900-pixel window, two
wrapped lines of it. A control that covers the page it is a control for is
worse than a control with one press too many in it.

> "That didnt work out how i hoped. this is too wide. […] make the tab buttons
> in the collapsed view constrained in width. i like the width of the original
> flyout. what i want is the ability to drag the tab buttons."

So the panel is a column in both states again — one branch in `placeNow`, not
two — and the strip is one line inside it that scrolls:

```css
.forge-assistant-workspaces {
    flex-wrap: nowrap;
    overflow-x: auto;
    overflow-y: hidden;
    touch-action: pan-x;
    overscroll-behavior-x: contain;
    scrollbar-width: none;
}
```

`nowrap` is what makes it a line. `overflow-x` is what keeps the rest
reachable. `flex: 0 0 auto` on the buttons is the half that is easy to lose:
shrinkable buttons fit eleven tabs into 360 pixels by making every label
unreadable, and then nothing overflows and there is nothing to scroll.

**`overflow-y: hidden` and `pan-x` are both about not stealing the page.** A
strip that took a vertical swipe would be a strip you could not scroll past on
a phone, and `overscroll-behavior-x: contain` stops a flick off the end
becoming the browser's back gesture, which leaves the page entirely.

**A mouse drags it.** Touch has a swipe and `pan-x` gives it to the browser;
a mouse has neither that nor — with the scrollbar hidden — anything to aim at.
`startStrip`/`moveStrip`/`endStrip` are the panel's own drag machine cut down:
the same six-pixel threshold, the same pointer capture, the same
clear-before-release ordering so a late `lostpointercapture` finds nothing to
undo. Four decisions in it matter:

*It declines every pointer that is not a mouse.* `touch-action: pan-x` means
the browser pans touch and pen itself and then sends `pointercancel`; running
both would fight the native gesture with a frame of lag and lose.

*A gesture that moved swallows the click that ends it.* Otherwise letting go
over a button switches to it, and a strip you cannot drag without changing tab
is a strip you cannot drag. The guard is a capture-phase listener on the row,
so the press is stopped before it reaches the button.

*The flag is cleared on the next press, not only by the click.* Let go between
two buttons and no click follows at all; a flag left standing would eat
somebody's next real press instead, which is the same bug one interaction
later.

*A keyboard activation is never swallowed.* `detail` is 0 for one, and it is
never the tail of a drag.

**The scrollbar is hidden and the cut edge is faded instead.** A horizontal
scrollbar under a tab row costs eight pixels of panel height to say something
the fade says in none. `markOverflow` writes `data-overflow` as `none`, `start`,
`end` or `both` from `scrollLeft` against `scrollWidth − clientWidth`, and the
stylesheet masks that side. There is deliberately no rule for `none`: when it
all fits there is no gradient at all, because a row that ends where it looks
like it ends needs no explaining.

**The header's Workspace menu goes while collapsed.** It and the strip are the
same list, and with the strip directly under it the menu is a press that buys
nothing. It is `hidden`, so it leaves the tab order too, and if it happens to
be open at the moment it is hidden it closes — a menu whose button is gone
cannot be dismissed by pressing that button again. Only that menu: the **⋯**
menu's button is still there, and closing it would be closing somebody's open
menu for them.

The resize handle comes back, since there is a width for it to set again.

**Measured in headless Chromium at 1280×800, driven with a real mouse**, because
none of this is arithmetic a stub DOM can check. Eleven tabs: panel 360 wide,
one line, 1338 pixels of row inside a 358-pixel window, no horizontal overflow
on the document, Workspace button hidden, resize handle back. A drag left
scrolls 280 and the fade goes `end` → `both`; dragging to the far end reaches
it and the fade goes `start`. A drag released squarely on *img2img* switches
nothing — and the very next genuine click on that same button switches to it
and closes the panel.

---

## 3.15 A launcher that stopped answering

> "I had one time webui showed me a disconnect message but i was still able to
> send image prompt request and see them render in gallery, but the fly out
> button would not respond. I could not switch tabs because i was in focus
> mode, so i had to reload my web page entirely. ... at one point in time i
> access my webui from a remote session before i saw the issue."

There were no logs from the occasion, so this started as speculation. It
stopped being speculation once one chain of events could be reproduced in a
real browser against the real scripts, and every link in it turned out to be a
defect in its own right.

### What the disconnect message was, and was not

Gradio 4.40 says "Connection errored out" or "Lost connection due to leaving
page" in a toast, top right, for a few seconds. It is not an overlay and it
covers nothing near a launcher; generation carrying on afterwards is ordinary
for Gradio, which opens a new stream per request. So the message was a
coincidence of cause, not the cause: whatever broke Gradio's stream -- a
machine asleep, a network changing hands, a remote-desktop session connecting
or disconnecting -- was the same event that broke things here.

### The launcher: a gesture that could not end

Reproduced in headless Chromium with the real scripts: a touch drag on the
launcher, a short slide past the threshold, and no touch-end -- which is what a
remote-desktop client delivers when its session drops or changes hands in the
middle of a press. Then three mouse clicks on the launcher. Nothing opened.

Three defects, each necessary:

1. **The drag preview took presses.** `.forge-assistant-ghost {
   pointer-events: none }` never applied: the root's own
   `#forge-assistant-root > * { pointer-events: auto }` is an id selector and
   outranks a class, and the preview is a direct child of the root. The element
   under the launcher's centre was measured as the preview, with computed
   `pointer-events: auto`.
2. **A touch drag could never end.** `endDrag` ignores every pointer but the
   one that started the drag, and every later touch has a new id. The drag -- and
   the preview over the launcher -- lasted the life of the page. With a mouse the
   next click happened to end it; a touch never could.
3. **Focus mode's only way out was the launcher.** Escape works, and nobody
   knows that.

And beside them, one of the same family: the edge resize handle added a window
`pointermove` and `pointerup` listener per gesture and removed them only on
`pointerup`, with no pointer id and no capture. A gesture that ended any other
way -- a touch the browser cancelled, which the handle's missing `touch-action`
made likely -- left them there, and from then on every pointer movement
anywhere on the page resized the panel and forced a layout doing it.

What changed:

- `supersede`, on `pointerdown` in the capture phase on the window: a new
  *primary* pointer going down means every earlier gesture has ended, whether
  or not its end arrived -- a mouse cannot press while pressed, and a new
  primary touch exists only once every finger has lifted. It ends them, and
  clears the click flags nobody consumed.
- Every gesture also ends on `lostpointercapture`, on the window losing focus,
  on the page being hidden and on `pagehide`; a mouse that moves with no button
  down is not dragging.
- The preview's rule is stated at the weight of the rule it has to beat.
- The resize handle is the same state machine as the other two gestures: one
  set of listeners for the life of the shell, a pointer id, capture, every way
  out reaching its end, placement once a frame, and `touch-action: none`.
- Only a drag of the launcher sets the flag that swallows the launcher's next
  click; a panel dragged by its header used to eat the next real press on it.
- **A safety net for whatever this list has missed.** Every listener is added
  through `Shell.on`, which now guards it: a handler that throws is logged once
  and `recover` runs, putting the shell back into a state it knows -- gestures
  ended, preview away, root back on the page if something took it off, exactly
  one of launcher and panel drawn -- while keeping what was *chosen*: open or
  closed, focus mode, the draft. And two presses on the launcher's box that land
  on something else, close together, reset the shell; if the launcher is still
  covered after that and focus mode is on, focus mode is left so the host's own
  tab bar is back. A dialog over the launcher is exempt -- it is meant to be
  there -- and what was lying over it is named in the console.

Measured after the change, same reproduction: the first mouse click opens the
panel and clears the stale drag.

### The conversation: two dead ends

Neither of these makes a launcher unresponsive, but both were on the same path
-- a network or process event while the page was away -- and both ended in
"reload the page".

**A stream that dies without closing was never replaced.** `silent()` could
always tell a dead stream from a quiet one; nothing asked it except a returning
tab, and even then it only flagged the stream, it did not let go of it. A
half-open connection -- a laptop lid, a VPN, a proxy that drops without a FIN --
is a `reader.read()` that never settles. A fifteen-second watchdog now asks, and
`restream` aborts the fetch and goes through the reconnect ladder. At the
moments things come back -- `online`, a page restored from the back-forward
cache, Chrome's `resume` after freezing a tab -- one missed heartbeat is enough
rather than three.

**A restarted server was the end of the conversation.** The server's key is
minted per process (`_token = secrets.token_urlsafe(24)`, "Regenerated by a
restart"), and the page reads it from a hidden Gradio field that nothing
redraws. After a restart every request is refused, and the page answered with
"Reload the page" every two seconds for as long as it stayed open; an event
from the new process's epoch was met with "The server restarted. Reload the
page to carry on."

Now a refused request renews the key from Gradio's own `/config` -- the
component tree Gradio serves the page from, behind Gradio's own `login_check`,
which is exactly as privileged as loading the page was. An endpoint of our own
would have handed the key to anyone who could reach the server, since routes
added to the app are not behind Gradio's auth. A renewed key means a new
process, so the session starts over from the bootstrap (`restarted`). Drafts
survive. Operations do not, and a send that was in flight is deliberately *not*
retried: the old process may have written it before it went, and a retry into
a registry that has never heard of it is how one message becomes two. Its text
is still in the draft, and the conversation that comes back shows whether it
arrived. A renewal that finds the same key is not repeated until something
succeeds -- the config is megabytes, and a refusal the key does not explain is
not worth fetching it every ten seconds for.

Measured end to end against a stand-in server that can go silent and restart:
the old store sat on the dead stream indefinitely and, after a restart, polled
with the stale key for ever; the new one replaces the silent stream after the
limit and not before, and rejoins a restarted server with one config fetch.

---

## 3.16 The performance review

> "Please review the overall implementation of the fly to ensure it does not
> introduce performance degradation to the webui. We want the safe options,
> please ask me first if you need to reduce our existing experience for
> performance."

Nothing had to be reduced. Everything that changed costs nothing visible.

**The finding that mattered: the navigation observer watched the whole
application.** To notice a tab switch it observed every `class` and `style`
change anywhere under `#tabs`, which is everything the WebUI draws. During a
generation that is every progress tick, preview frame and status flip, and each
one re-read every tab -- with an ancestor walk reading computed styles per tab to
ask whether it could be focused, which nothing on that path ever used -- called
every listener, and rebuilt the collapsed panel's tab strip. Measured in
Chromium over five seconds of simulated generation with the strip open:

| | before | after |
| --- | --- | --- |
| active-tab lookups | 606 | 2 |
| full tab descriptions | 909 | 0 |
| listener calls | 303 | 1 |
| strip rebuilds | 303 | 0 |
| main-thread time in them | 330 ms | ~0 ms |

-- the two and the one being the real switch at the end, which is still seen.
What is observed now is exactly what decides the selection: each top-level
panel's own attributes and the top-level bar's buttons, with a third observer
rebinding them when panels come and go. Listeners hear only a change they could
show: the active tab, the pending switch, a button renamed or disabled. The hot
paths use a pairing that reads no styles; whether a tab can be focused is asked
lazily, when something reads it. The strip is rebuilt only when its tabs change,
and otherwise re-marked, so a keyboard focus or a drag on it survives a switch.

**A streamed reply redrew the panel on every token.** Renders from the store are
now drawn once per animation frame with the latest view: markdown, fingerprint
and a layout read, once a frame instead of several times a frame, and not at all
in a hidden tab until it is looked at.

**The resize handle forced a layout per pointer event**, and leaked (§3.15). It
places once a frame now.

What was looked at and left: the three window `pointermove` listeners (a
comparison each and a return), the placement on `visualViewport` scroll (needed
to follow a pinch; once a frame), focus mode's bounded walk on entering (once
per toggle), the store's fifteen-second watchdog, and the focus mode resize
observer (a no-op for every workspace but those with a canvas to refit).

---

## 3.17 Customize

> "In the '…' menu, add a customize menu. It should open a pop up with
> customize options ... Background color, title, corner radius, font color,
> stroke, the same animation that we have for our custom progress bar
> animation always playing on the button ... Cool things that are exciting and
> can catch your eye because now this the primary controller."

`forge_assistant_look.js`, on its own so that it costs nothing until opened and
can be tested alone. A **look** is a complete, validated set of choices: title,
icon, size, text colour (or Auto), fill (or a two-stop gradient and its
angle), border colour and width, radius (or a pill), glow, the colour effects
are drawn in, an animation and its speed, and whether to animate despite a
reduced-motion preference. Presets are looks.

**Stored, so checked.** In `localStorage`, per browser, like Free Float. On the
way back in every field is held to what it can be: colours are six-digit hex or
nothing -- not a name, not `url()`, not `var()` -- numbers are clamped, choices
are checked against their lists, the title loses control characters and is cut
at forty characters (by character, never half way through one), and a version
this build does not know is the default rather than a guess. Nothing from
storage is ever written as markup.

**The animations are the progress bar's.** Sheen, Pulse, Neon and Ooze -- Ooze's
bubbles are `.mc-ooze-bubble` on `mc-ooze-rise`, the progress bar's own class
and keyframes, not a copy that can drift -- plus Aurora, a conic rainbow turning
behind a cover inset by the rim's width. *Match progress bar* reads
`model_chain_style_theme` (and the Custom toggles) at the moment the look is
applied, and draws in the progress bar's colour. **Every keyframe animates
transform and opacity only** -- the sheen band translates, the halo's pre-drawn
shadow fades and scales, Ooze's surface skin is a strip twice as wide translated
by its own period, Aurora rotates -- so the compositor runs them with no layout
and no paint, and a test holds every `fa-look-*` keyframe to that. They stop
when the launcher is not displayed.

**Legibility is checked against what is really behind the title.** The first
Toxic preset put lime text on a dark fill, and Ooze then painted bright sludge
over the fill: 1.4 to 1. The contrast check, and Auto, now measure against the
sludge for Ooze; the dialog warns under 3 to 1, and a test holds every preset to
4.5.

**The dialog** is a native modal `<dialog>`, built on first use: the browser
supplies the backdrop, focus containment and the top layer, which is above focus
mode and above another extension's dialog layer. The panel steps back to its
launcher first, because the launcher is the thing being customized and it takes
each change live behind the backdrop. Escape is the dialog's cancel -- the shell's
own Escape handling stands aside while it is open, where it would otherwise have
left focus mode behind the dialog. Verified in Chromium: no control under 44
pixels, eight live previews, the phone sheet exactly the screen.

## 3.18 Focus takes the browser full screen too

> "When focus mode is turned on for the flyout, the tab bar goes away, but i
> would like for this to trigger the browsers to go into full page view as
> well. One press of the toggle = tabbar hidden & browser goes full screen
> view... press toggle again = tabbar returns & exit full page browser view."

Focus was built as deliberately *not* the Fullscreen API -- the header of
`forge_assistant_focus.js` said so, and so did the README: full screen
takes the browser's own chrome as well, it needs a gesture every time, and an
element in full screen is the only thing the browser draws. The first of those
is now what is asked for, and the other two decide how it is done rather than
whether.

**The document goes full screen, never the workspace.** Focus itself is
unchanged -- it is still the transaction in `forge_assistant_focus.js`, a class
on and a class off. What the toggle adds is `document.documentElement` full
screen: the page with the browser's bars taken away, inside which focus works
exactly as it does without, and the assistant, the dialogs and the toasts are
all still drawn. The workspace in full screen would have taken the assistant
off the screen, and with it the way back out.

**Inside the press.** A browser grants full screen only to a page somebody has
just interacted with, so the request is made from the Focus toggle's own click,
after focus has been entered successfully -- a refused focus asks for nothing.
A reload cannot bring full screen back, which already matched focus: a reload
has always started outside it.

**One press, both ways, and never out of step.**

| What happens | Focus | Full screen |
|---|---|---|
| Focus pressed on | on | asked for (`navigationUI: "hide"`) |
| Focus pressed off, or Escape reaches the page | off | ended, if it is ours |
| The browser ends it: Escape, Android's back gesture, a switch of tab | turned off to match | already gone |
| A switch of workspace inside focus | off and back on (§3.11) | kept |
| A workspace that cannot be refocused, a covered launcher | off | ended, if it is ours |
| A video made full screen on top | on | the video's, then ours again |

Every path that turns focus off now goes through one function, `focusOff`,
because there were five of them writing the same three lines and a sixth
concern -- the full screen -- would have been forgotten by at least one. A test
holds the source to a single `focusEnabled = false`.

**Only its own.** Something already full screen when focus is turned on -- the
page made full screen by something else, a video -- is neither claimed nor
ended when focus goes off. F11 is not full screen as far as the API is
concerned, so it neither blocks the request nor is ended by the exit. A video
stacked on top of our full screen is left playing when focus goes off.

**Every answer the browser can give.** Granted: ours. Refused (a frame without
`allowfullscreen`, a setting): focus stands on its own, and the next press may
ask again. Granted after focus has already gone off -- two quick presses -- the
screen is given straight back. Granted and already over by the time it is
read: not left pending. Never answered at all: not waited on after five
seconds. Another element's refusal: not ours, although the error event bubbles
to the same document. Safari's prefixed names -- iPadOS, and Safari before
16.4 -- are enough on their own.

**Where there is none.** iPhone Safari gives full screen to videos and nothing
else. There the toggle does exactly what it did before, and says nothing about
it.

**The other extension on the page.** Mini Paint's View Outputs player took any
full screen for its own: with the page full screen, its button "left" the
page's instead of entering the stage's, and closing the view or picking another
output ended the page's -- and with this change, focus mode with it. Fixed on
Mini Paint's side in the same round: its full screen is its stage's, and
stacked on the page's it comes off by itself.

Verified in Chromium with real presses: one press, the page full screen and
the tab bar gone; a second, both back; the page's full screen ended by script,
focus off; a switch to img2img inside focus, still full screen; a stage made
full screen on top and taken off again, still focus and still ours; the real
Escape key, both off.

---

## 3.19 The second round: read aloud, tap to reveal, Send to prompt, Send again, Auto Attach

> "The goal: When conversation mode is open, I want an option to enable the
> current visible image in the text to image or image to image tab ... to be an
> automatic input to the next LLM prompt ... Make this a toggle in the '...'
> menu 'Auto Attach' ... the buttons are always there for EDIT, RETRY, and
> DELETE. I want these buttons to only show up when i tap on the reply ... a new
> button there for SEND PROMPT ... a new button for SEND AGAIN ... the TTS mode
> seems to not turn off in flyout mode. The button does not change state when i
> click it, and i seems to always see TTS activity in my console."

Asked for together, as a "V2 flyout menu update". Three decisions were put to
the user before any of it was built, and accepted as proposed: Auto Attach does
not attach the same picture twice while the thread still carries it; Send again
on your last message answers in place, in the tab as well; Send to prompt writes
to txt2img from any tab that is not an image tab.

### The read-aloud switch was three defects

1. **It never asked.** `aria-checked` started `false` whatever Voice Chat's
   *Speak replies automatically* said, and nothing ever set it from the
   setting. With speech on, the first press "turned it on" and changed nothing.
2. **It wrote by pressing somebody else's control.** `setAutomaticReadAloud`
   clicks the LLM Studio checkbox. Where that press did not land, the setting
   did not move.
3. **Nothing drew it.** No rule in `style.css` named its state, so on and off
   looked identical -- the report, word for word.

Behind them, the cost the user was seeing: with the setting on, the acceptance
path (`mc_llm_conversation_ops` calling `mc_voice_ui.begin_speech`) creates a
speech turn for every reply, and a turn warms the voice worker and synthesises
from the first segment whether or not a page ever opens its stream -- it gives
up only after `mc_voice_turn.CLIENT_WAIT`, thirty seconds.

What replaced it:

- `bootstrap()` carries `read_aloud` -- the setting, or `None` when it cannot
  be read, because "off" is a claim. The store keeps it (`noteReadAloud`, on
  both bootstraps) and every render draws the switch from it: `data-state` on,
  off or unknown; a lit speaker, a greyed speaker struck through, a plain
  speaker. Off is greyed and deliberately not faded: a faded control reads as a
  disabled one.
- `POST …/v2/read-aloud` writes it. It shares `mc_voice_ui.apply_auto_speak`
  with the tab's checkbox, so off cancels the turn being spoken from either,
  and it answers with what the store holds, so a refused write reads as the
  switch staying put. It refuses anything but a boolean: `"false"` is truthy.
- The store moves the switch at once and puts it back if the route refuses.
- **The part that makes off mean off:** `Store.stampVoice` writes
  `voice: false` into every GENERATING envelope the panel sends while the
  switch is off, and the server then never calls `begin_speech` for that
  reply. Written once per envelope and never rewritten, because a retry is the
  same envelope and one operation id arriving with two payloads is refused as a
  different request.
- Off also silences the page on the press (`speech.stopPlayback`), and the
  facade's `setAutomaticReadAloud` is still called afterwards -- with what the
  server stored -- so the tab's checkbox agrees.

Found on the way and left alone: `capabilities().voice` calls
`mc_voice_state.enabled()`, which does not exist, so it is always `False`.
Nothing reads it.

### Tap to reveal

`ACTIONS` is a table with `newest` and `role` on each entry, so which message
gets which icon is one list rather than a chain of conditions: Edit, Regenerate
and Delete on the newest message (Regenerate only on a reply), Send to prompt
on every reply, Send again on your own newest message. A reply still being
written offers nothing.

The row is built `hidden` and shown by `reveal(key)`. Four details:

- **Delegated.** One click listener on the transcript (`tapBubble`), because
  bubbles are rebuilt whenever the thread moves. A press on a link or a control
  inside the bubble is that control's; a selection inside it is reading.
- **Kept by key, re-applied after a redraw.** A reply arriving token by token
  redraws once a frame, and a row that closed on every token could not be used
  while a reply was arriving. A key no longer on the page is forgotten.
- **Put away before a press lands.** `dismissActions` runs on the window's
  `pointerdown` in the capture phase, beside `supersede`: a press on another
  message closes this one there and opens that one in `tapBubble`.
- **Keyboard.** A bubble with actions is focusable, and Enter and Space toggle
  it.

Driving it in Chromium found the one layout problem in this round, and it was
older than the round: the panel is placed from its own height, and nothing
placed it again when the height changed. Docked along the bottom it grew
downwards off the window as messages arrived -- and now as a tap opened a
message's actions -- taking the composer with it (bottom at 993px in an 800px
window). A `ResizeObserver` on the panel places it again when its height
changes; width is ignored because `placeNow` writes the width itself.

### Send to prompt

`keptFromPrompt` reads the prompt with `prompt_master/krea/literals.py`'s
grammar -- `[[…]]`, a sign only immediately before `[[`, the first `]]` closes,
an empty command carries nothing, an unclosed one is ordinary text -- and with
`extra_networks.KINDS` (`lora`, `lyco`, `hypernet`, any case, `[^<>]*`) for tags
outside a command. What it keeps, it keeps verbatim and in source order; the
new prompt is the trimmed reply, a newline, and those joined by spaces. It is a
JavaScript copy of rules that live in Python, so its tests pin it to the same
cases the Python modules are tested with.

It writes the textarea inside `#txt2img_prompt` or `#img2img_prompt` and tells
Gradio through Forge's `updateInput` where it exists (an `input` event where it
does not) -- the pattern `model_chain_spatial_krea.js` already uses. Nothing is
pressed.

### Send again

`_plan_resend` answers a *last* message in place: `APPEND` at
`len(messages)`, the history as it stands, `persisted=False` and the current
revision as the accepted one -- the shape Regenerate's final-reply path already
has. The completion's own check (`index == len(messages)`) and the revision
guard keep it honest. Earlier messages still branch (S8). The tab's *Send again
from here* goes through the same planner, so it changed the same way, and
`_follow_thread` already handled a result that lands in the thread it started
in.

### Auto Attach

Which picture is Forge's own answer -- `extract_image_from_gallery` takes the
selected thumbnail, else the first -- read off Gradio 4.40's gallery in order:
the preview's `detailed-image`, the selected thumbnail, the first thumbnail,
and, for a theme that renames all of those, the first `img` that is not inside
Forge's `.livePreview`. The picture is fetched from its `/file=` address and
goes through `Store.upload`, the paperclip's route, so the server stages it
exactly as a picture attached by hand.

The rule against sending the same picture twice is "the same `src` as the last
one Auto Attach sent in this conversation, *and* the thread still has a message
with a picture". The second half is what makes deleting that message enough to
have it attached again. Formats staging refuses, and pictures past its size
limit, are redrawn onto a canvas as JPEG, no more than 2048 pixels on the long
edge.

A send with Auto Attach waits for the upload, and Send is disabled meanwhile by
a rule it already had (an attachment that is not ready cannot be sent). If the
selection moved during the upload, nothing is sent -- the picture stays in that
conversation's draft.

### The status line holds

`say` is overwritten by the next render, and a press that changes the store
causes one, so the answer to a press was on screen for a frame. `tell` holds the
line for six seconds over the idle sentences, and a *warning* over the progress
of a reply -- the press that left a picture out is usually the press that
started the reply. An error from the server and a lost connection still take the
line at once.

### Verified in Chromium

Against a stand-in server, with the real scripts and stylesheet, a Forge-shaped
tab bar and a gallery built the way Gradio 4.40 builds it (nested buttons and
all), driven with real clicks. Twenty-two checks: the switch starts from the
server's answer, writes through the route and looks different in each state;
there is no action row before a tap, a tap shows it, and a second tap or a
press elsewhere hides it; Send to prompt writes the expected text, fires
`input` and sends no command; Auto Attach lights the paperclip, uploads the
gallery's PNG and sends its token with `voice: false`; the picture shows on the
message; your last message offers Send again, which sends `resend_from_user`
at the right index; the panel stays on the window as it grows and as a row
opens; the same picture is not attached twice; no script error. It is not a
Forge: the gallery's shape is from Gradio's source, and the first run on the
real host is the gate still open.

## 3.20 The third round: New thread, the system prompt editor, and four repairs

> "1. Add the ability to 'start a new thread' directly from the fly out menu
> ... 2. ... I like the idea of a full page editor for system prompt, with
> option to restore default ... in LLM studio character menu, i just want a
> button that opens a full page editor like this ... For flyout, put this
> option to open this view in the '...' menu 3. ... if i hit 'enter' ... the
> prompt stays in the input field ... 4. The height of the flyout conversation
> mode is too tall. Lets reduce it by 1/3 ... 5. ... the 'image attached' and
> 'on state' for TTS ... looks bad and low contrast. 6. ... the image is not
> attached, and its name is listed twice ... I should never see the file name."

Six clean-ups asked for together, each its own commit. The reference for the
editor was Mini Paint NEO's system prompt editor, without its model and
enhancer choices.

### New thread

`Store.createThread` sends the service's own `create_thread` -- the command the
tab's New thread sends -- with `expected_revision: null` (there is nothing yet
to compare against) and an empty payload, then selects the thread it made. So
the thread is named and greeted exactly as from the tab, and the panel moves
while the tab stays put, as a thread chosen in the panel always has. The item
opens a collapsed conversation, since whoever asked for a thread is about to
write in it, and is disabled until the panel is on a conversation.

The menu reading the selection broke two Customize tests whose stand-in shell
had no store. They failed before asserting anything; the setup gained the stub
and the tests kept their intent. It was found when the whole suite ran, which
is the argument for running it.

### The system prompt editor

**What it edits.** The character's `system` field, the override LLM Studio's
character editor already writes and `prompt_master.chat.prompt.system_text`
already reads. One prompt per character and two places to change it, not two
prompts. `GET …/v2/system-prompt?character=` answers `{character, text, source,
default}`: `text` is the override exactly as stored (`{{char}}` and all) or the
prompt built from the Context and the persona -- built with `ops._persona()`,
the function replies use, so the default shown is the one a reply gets. `POST`
takes `{character, text}` or `{character, restore: true}`.

**The rule it adds.** Text that *is* the built prompt, give or take the white
space at its ends, is not an override, and neither is empty text. The editor
opens on the built prompt, so Mini Paint's semantics -- Apply saves whatever is
in the box -- would have turned a press on an untouched page into a frozen copy
that stops following the Context and the persona, with nothing on screen to
say it had. Mini Paint's defaults are shipped text and cannot drift; a
character's are built and do.

The rest is the service's usual shape: an unknown character is `NOT_FOUND`
(404), no character or a missing `text` is `INVALID_INPUT` (400) -- missing is
not empty -- a prompt past `MAX_TEXT_BYTES` is refused rather than cut, a
failed write is `SAVE_FAILED` and retryable, and `restore` must be a real
boolean for `read_aloud`'s reason: `"false"` is truthy. The same text applied
twice writes once. Nothing is published: `character_changed` means "the tab
moved to another conversation", and a panel told that would follow it.

**Why it is not a Gradio panel, as Mini Paint's is.** Mini Paint's editor lives
in its own tab and is a column the stylesheet promotes to fill the window. This
one has to open from the flyout on any workspace as well as from LLM Studio, and
a Gradio component belongs to one tab. So it is one DOM editor,
`javascript/forge_assistant_system.js`, and LLM Studio's **⤢ System prompt** is
a js-only Gradio event (`fn=None`, the dropdown as its one input) that hands it
the character -- no round trip through the queue. The flyout's item is the same
call.

**How it behaves.**

- A native modal `<dialog>`: the top layer, above focus mode, the panel and
  Mini Paint's dialog layer, with focus contained and Escape arriving as
  `cancel`. It is appended to the body and says `role="dialog"
  aria-modal="true"` out loud, because focus mode's path rule spares what
  carries those; in Chromium the path rule never reaches the body's own
  children anyway, since the body is not its own descendant, so the attributes
  are the second line, not the first. The shell's `documentKey` stands aside
  while it is open, as it does for Customize.
- The dialog's own `display` is never set -- an author `display` beats the
  browser's `dialog:not([open])` -- and the column is its sheet's.
- Apply and Restore are not pressable until the prompt has been read. An empty
  box applied is "restore the default"; a failed read must not become a press
  that wipes an override. Reload always is.
- Close, **×**, Escape and Reload ask (`confirm`) while the box holds something
  unapplied; typing says *Not applied yet.* A second press on the way in, on
  the same character, keeps the edit rather than re-reading over it.
- A counter drops answers for a page that has moved on: a read that lands after
  the editor was closed and opened on someone else is not drawn. A save that
  lands then still reaches LLM Studio, because the server has it.
- When LLM Studio's editor is open on the same character (its Name box says so,
  compared without case), the override box is written and told with
  `updateInput`, the way a person typing would -- or its next Save writes the
  old prompt back. Its preview follows, because that box's `change` rebuilds it.
- Names go on the page as text, never as markup: a character is called what its
  file says, and a file can say anything.

**Found in Chromium: the phone layout.** On a 412 by 915 phone showing a page
508 pixels wide, the browser widens the *layout* viewport to 508 by 1129, and
`position: fixed` -- the top layer's too -- is placed in that. The editor filled
it: its right edge was off the glass, focusing its **×** panned the view 96
pixels sideways, and its buttons were 200 pixels below the fold, while every
check that compared it with `innerWidth` passed. It is now fitted to the visual
viewport when it opens and refitted on the visual viewport's `resize` and
`scroll` while open, which is also the viewport a phone's keyboard shrinks. The
check now compares with the visual viewport and asserts the page really is
wider than the phone.

### Enter left the message in the box

`render()` never rewrites a focused input -- it would fight the person typing --
and Enter leaves focus where it is, so after an Enter the draft was cleared in
the store and the words stayed on screen (a press on Send moved focus, which is
why only Enter showed it). `emptyComposer(key)` runs after an accepted send and
empties the box only when that conversation's draft is empty: words typed while
the message was sending are a new draft and stay, and a refused send keeps its
text to be sent again.

### A third shorter

The panel's chrome -- header, selector, status line, composer -- measured 268
pixels at every window size tried, so the transcript alone gives up the third:
`max-height: max(120px, calc(32vh - 89px))` in place of `48vh`. In Chromium the
open panel went from 652 to 435 pixels at 1280 by 800, from 787 to 525 at 1920
by 1080, and from 708 to 472 on a 412 by 915 phone -- two thirds, each time.
Scrolling and following the newest message are unchanged.

### The lit buttons

Both lit states filled from `--color-accent-soft`, which the Lobe theme sets
nearly white: a white paperclip button under a dark theme, and a pale speaker
the glyph could barely be seen on. The earlier fix to LLM Studio's transcript
records the same property doing the same thing. One rule now draws both: the
accent as border and inset ring over a 22 % tint of the accent in the button's
own neutral surface (`color-mix`), so the glyph keeps the contrast it has when
off. Measured: Lobe dark, lit (46, 42, 77) against (28, 28, 28) off, ring
(109, 93, 252); light, lit (198, 212, 243) against (244, 244, 245). The unread
count on the launcher had the same fill and has the same cure at 35 %. A test
holds that no rule of the panel's fills from the soft accent.

### The picture and its name, twice

The served-picture route requires the page's key, and an `<img src>` cannot
send a header, so every picture was refused. The browser then drew the
image's alternative text, which was the file name, above a `figcaption` that
was the file name again -- the screenshot's "tmpsocax_1o.png" twice.
`Store.picture` now fetches with the key and hands the image a blob URL,
remembering the last 48 and revoking what it lets go; a failure is not
remembered, so it is asked again next time. The figure has no caption, its
alternative text is "Attached picture", and a picture that is unavailable or
cannot be fetched or decoded is a placeholder, "Picture unavailable". Nothing
on it carries the name.

### Verified in Chromium

Against the stand-in server, extended with a picture route that refuses a
request without the key, the system prompt route and `create_thread`, and a
page carrying LLM Studio's Name and override boxes and the ⤢ button run the way
Gradio runs a js-only event, from the Python module's own string. Thirty-three
checks: the picture is fetched with the key and decoded, the placeholder shows,
no file name anywhere in the transcript's text or attributes; Enter sends and
empties the box; the menu reads Free Float, Auto Attach, New thread, System
prompt…; the editor is modal, covers the window without reaching under the
scrollbar, is mostly its box, lands focus on **×**, opens on the default,
keeps no override when the default is applied untouched, saves an override and
writes LLM Studio's box with an input event, asks before Close throws an edit
away, restores the default and empties LLM Studio's box, closes on Escape and
is then off the page, opens from LLM Studio's button on the character, stays
on top and pressable in real focus mode and leaves focus mode on when its
Escape closes it; New thread sends `create_thread` for the character and moves
the panel onto the greeting; on a phone the editor is the glass, buttons on
it, text 16 pixels; no script error. The phone was Chromium's emulation, not a
phone.

---

## 4. Deliberate deviations

**The panel offers a few message actions, not all of them** (§3.11, §3.19).
The specification's shared conversation view carries every action the tab has.
At the user's request the panel carries edit, regenerate and delete on the
newest message, Send to prompt on replies and Send again on your own newest
message -- as icons, and only once the message is tapped. The service implements
all sixteen and the tab offers them; this is one view's opinion about what
belongs on a phone-width panel, not a change to the conversation. The cost is
that alternate versions from a regenerate are browsable in the tab and not in
the panel, which was accepted when it was asked for.

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
| G2 Theme mount points, portal roots, listener order | **closed for Lobe, from its source (§3.8).** The theme's mount point is `<gradio-app> > #root`, Gradio's container is moved inside it, and its header is a sticky sibling of the content — an unmarked child of a marked ancestor, hidden by the path rule. Tested against a DOM of that shape. Listener order remains unverified against a running host. |
| G3 Installed tab ids and header controls | **closed for Gradio 4.40, from its source (§3.8).** The adapter reads `#tabs > .tab-nav` (the shallowest bar) and `#tabs > [id^="tab_"]`, pairs them by `aria-controls`, and ignores nested tab groups. It never matches a tab by display name. |
| G4 Whether `script.js` honours `defaultPrevented` | **open.** No document-wide Send or Stop shortcut is installed, so nothing depends on the answer. Escape uses a precedence function with its own tests; the one thing it will not do is swallow Escape when nothing closer to hand wants it. |
| G5 Hidden field value before its tab is selected | **open.** The capability is rendered into `mc-llm-chat-conversation-key` exactly as `mc-llm-chat-voice-key` is, which is the pattern already proven in this installation. If it turns out to be empty before the tab is selected, the fallback is a bootstrap route behind the host's own auth; the store already treats a missing key as "conversation unavailable" and keeps navigation and focus working. |
| G6 Whether generation can run outside a Gradio worker | **closed enough to ship, in this checkout.** `mc_llm_sessions.conversation()` takes a request and a cancellation and yields events; the tests drive it from a plain thread. If a host turns out to need request context, the executor is the one place an adapter goes. |
| G7 Gradio image input via DataTransfer | **not needed.** See §4. |
| G8 `gr.Chatbot` row classes | **closed for Gradio 4.40, from its source (§3.10).** The three speculative selectors were the defect, not the risk: they constrained `.message`, the text box *inside* the bubble, so the bubble was never limited and the text wrapped at three quarters of its own box. The geometry is now stated against the real shape — `.message-row`, `.flex-wrap`, `.flex-wrap > .message` — with both halves keyed on `.flex-wrap` so a rename stops them matching together and leaves the component's own layout rather than half of ours. |
| G9 Base path and secure context | **partly closed.** The base path is read from the document's own URL and the routes are registered under it, with a test. Secure context is a deployment fact: without HTTPS there is no microphone, and `voice_chat.js` already degrades cleanly. Image paste works either way. |
| G10 Filesystem atomic replace and fsync | **closed for the write.** `atomic_write_json` fsyncs the file and renames; `mc_llm_conversation_store.fsync_directory()` is there for the rename, is never fatal, and is a no-op on hosts that do not support it. |
| G11 Test environment | **closed.** `pip install pytest pillow numpy httpx` and the suite runs. Baseline at `78da206` was 6,113 passing, 13 skipped, zero failures; this work leaves it at 6,491 passing, 13 skipped. |
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
