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
| `javascript/forge_assistant_layout.js` | the results column fills the window, evenly spaced, with a grid of whole rows (§3.33) and room for what a progress bar paints above itself (§3.35); Txt2Img's settings column docked in the panel (§3.29), its fields engaged by a tap (§3.31) |

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

It did not apply to the phone sheet, which is anchored to a half of the screen
and covers it, and the preference waited for a wider window. That was reversed
at the user's request (§3.30): with Free Float on, a phone gets a floating
panel too.

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

## 3.21 The fourth round: editing in the panel, checked switches, Send to Generate

> "This is what happens when i try to edit a prompt in the flyout view. Please
> update so that the input field becomes the normal input box ... make sure it
> feels clear that I am editing a previous message, not simply submitting a new
> one ... That image is what my browser threw me to edit. It sucks. I dont want
> the browser doing this, i need UI in our flyout ... i want the menu items to be
> state wise ... I want a toggle that enhances the "send to prompt" with a "Send
> to Generate" ... pressing the button under a reply would send the prompt in to
> replace the current, and invoke a generation as if user pressed the button."

Asked for together as a V4. Two choices were left to this side and one pair of
defaults was put to the user and accepted: edit in the composer rather than in
the bubble; a check beside a switch that is on; Send to Generate goes to txt2img
from a tab that is not an image tab, and does not queue behind a run.

### Editing in the panel

`startEdit` was `window.prompt`: one line of the browser's chrome, no room for a
long prompt, and on a phone a dialog over everything. It is now an editing state
of the composer.

**Why the composer and not the bubble.** Bubbles are rebuilt whenever the thread
moves (`renderTranscript` keys them by epoch, revision and index), and an editor
inside one would have to survive every redraw with its caret and selection; the
transcript is a third of the window since §3.20, too little room to edit a long
prompt. The composer already has the keyboard -- a phone's included -- and the
growing, and it is where somebody expects to type.

**How it says it is an edit.** Four things at once, all from `--color-accent`
and never from the soft accent (§3.20's lesson): a strip above the box naming
the message (*✎ Editing your message*, *✎ Editing Ada's reply*) with Cancel; an
outline and glow on the box; the same outline on the message in the thread,
re-applied after every redraw (`markEditTarget`); Send reading Save in the
theme's primary colours. Measured under the Lobe palette in Chromium: the
strip's text about 12:1 on its tint, Save's white on the accent 4.54:1.

**What it keeps apart.** While the box holds an edit it is not the draft:
`typed()` -- the input handler, named so it can be tested -- does not write the
store's draft, the ordinary draft sync in `render` stands aside, and
`finishEdit` puts the conversation's draft back. Moving to another conversation
ends the edit and shows *that* conversation's draft, even with the box focused,
where the ordinary sync would leave the old one on screen to be typed into the
new thread. The save's envelope names the conversation the edit began in, so a
selection that moves before the redraw cannot re-aim it.

**What Save does.** `edit_message` with the index, version and revision the
Edit was pressed on, `image_action: "keep"` -- the words change and nothing
else, as `_edit` has always done; no reply is asked for. An unchanged edit
sends nothing and ends; an empty one is not saved; a second press while saving
sends nothing more; a refusal (or a request that throws) leaves the edit in the
box with the reason and "Your edit is still in the box." The paperclip and
microphone stand aside, and a pasted picture is refused rather than staged into
a draft nobody can see. Escape cancels only when focus is inside the panel: an
Escape in Forge's own prompt box is that box's.

**Found in Chromium.** Send is disabled over an empty draft, and the edit
inherited that: Save was greyed out over a message full of words until
something was typed. The edit now sets Save's own state when it starts.

### Checked switches

Free Float was drawn with an accent border when on and Auto Attach with nothing.
One rule now draws every `menuitemcheckbox` in the ⋯ menu: a check in a column
of its own, from `aria-checked`, so what a screen reader hears and what is on
screen are one fact and a later switch gets it by being a switch. It is
generated content -- the label's text stays the label -- declared as `"\2713"`
and then `"\2713" / ""` (decoration to a screen reader), in that order, so an
engine that cannot read the second form keeps the first. Every entry keeps the
column, so the labels line up whether a switch is on or not.

### Send to Generate

A switch after Auto Attach, off by default and remembered in `localStorage`
under its own key. Off, nothing changes. On, the reply's ➤ is drawn as ▶︎
(U+25B6 with U+FE0E, so it stays a glyph and not a coloured emoji) named Send to
Generate -- bubbles are kept across renders, so `renderSendMode` redraws the
ones on the page when the switch moves -- and a press does four things:

1. **Where.** The image tab you are on; from any other, txt2img, reached with
   `switchWorkspace` first (focus mode comes along). A switch that fails still
   generates, and the status line says where.
2. **What.** The prompt exactly as Send to prompt writes it: `promptFrom` keeps
   LoRA tags and literal commands, written with Forge's `updateInput`.
3. **When.** Gradio reads a component's value from its own state, not the DOM.
   The press waits two animation frames (or a quarter of a second in a tab the
   browser is not painting), then checks the box still holds what was written,
   and only then clicks.
4. **Not twice.** Forge's `setSubmitButtonsVisibility` shows Interrupt and Skip
   over Generate while a run is on, and Interrupting... while one is stopping.
   Any of those showing (computed style: idle, the stylesheet hides them), or
   Generate itself hidden or disabled, and nothing is pressed; the prompt is left
   in place and the status line says to press Generate when the run finishes.
   Asked once, at the moment of the press -- a run can start during the wait.

The press is `generate.click()` on the real button, so what Generate does with
the page -- the pipeline, the literal boxes, everything -- happens because it
is Generate. Nothing in this extension had pressed it before.

### Verified in Chromium

Against the stand-in, extended with Forge's generate box (Interrupt, Skip and
Interrupting hidden until a run, shown for a second and a half after a press),
a Gradio stand-in that takes a prompt's value a frame *after* its input event
and whose Generate reads only that, working tabs, and `edit_message` on the
server. Twenty-seven checks: Edit opens no browser dialog, puts the message in
the box and marks it every way, Enter saves `edit_message` with
`image_action: "keep"` and asks for no reply, the thread shows the new words,
the draft comes back, Escape cancels, a forty-line edit gets the 292-pixel box
and the panel stays on the window; no switch is checked while off, the labels
line up, the one switched on is the one checked; the reply's button says Send to
Generate, Generate is pressed once with the kept LoRA and literal, a second
press during the run presses nothing, from Extras it switches to txt2img and
generates there, on img2img it generates in img2img, and off again the button
only writes; on a phone the strip and Save are on screen and Save is pressable
from the start; no browser dialog and no script error anywhere.

Two things it is not. The stand-in's Gradio is a model of the real one's
timing, not the real one: the first run against Forge Neo is the gate still
open for Send to Generate. And the stand-in's page had to be kept from
overflowing a phone's width for the phone checks: on a page wider than the
phone the browser widens the layout viewport, and the panel's bottom sheet --
positioned against it, as it was before this round -- sat below the visible
screen until the page was panned. That is older than this round and was left
as it is.

## 3.22 The fifth round: glyphs, Chat, three presses, a dialog to edit in, a grouped menu

> "I want the line that has the character name and conversation name removed
> ... the accordian for conversation i want it removed. Instead, i want a
> simple button in the top row that reads Chat ... remove the Focus button.
> Instead, i want triple tapping on the drag area to be the toggle focus on /
> off ... I want the buttons rows to have reduced height ... convert all
> buttons with text to icons ... when i choose to edit, i want a simple pop up
> ... current text in an input field, cancel, and a done button ... For both
> LLM studio conversation and our flyout menu ... shift + enter to add new
> lines ... should not be full screen ... responsive ... so it also looks
> good on mobile ... add a function to the menu from conversation mode New
> Chat ... reorganize the '...' menu, maybe add some nesting."

Six asks, and the same shape as the rounds before: the panel's chrome gets out
of the conversation's way. One recommendation was made and taken up -- the
shape of the grouped menu -- and one thing was traded away knowingly, below.

### The header, and what became of Focus

One row still, and every control in it a glyph: 📑 Workspace, 💬 Chat, ⋯, the
grip, ✕. `iconButton()` makes each with its word as `aria-label` and `title`
and never as text, so the row is as tall as its glyphs and a screen reader
hears the word. Chat is the accordion heading moved into the row: it carries
the heading's `aria-expanded` and `aria-controls`, `applyChat` (the old
`applyAccordion`) writes its state, and the stylesheet lights it while the
conversation shows. The heading and the selector line -- "Prompt Bot · New
chat", also the thread list's button -- are gone from the DOM and the
stylesheet; where the name went is in the menu, below.

Focus has no button. Three presses on the header's space -- `headerClick`
counts a click whose target closes over no control, `tapHeader` keeps the
count for `TAP_WINDOW` and acts inside the third press, for the launcher's
reason (a browser grants full screen only to a press still being handled).
One press and two do nothing, so nothing waits on the window; the count is
forgotten when it passes. What the launcher's counter learned the hard way is
carried over: the tail of a drag is not a press (`endDrag` sets
`suppressHeaderClick` for a panel dragged by its header; `startDrag` clears it
at the next press, because with the pointer captured the click may land on the
panel and never come), and `cancelGestures` forgets a half count. Focus state
is written on the panel (`data-focus`, `markFocus`) for anything that draws
from it; nothing does yet.

Both rows are 36 pixels, not 44: the header's and the composer's buttons, the
input, Send (➤, the theme's primary pair) and Stop (■). A list's entries -- the
menus, the actions under a message -- keep the full 44, because a list is what
a thumb picks from and a row is chrome.

### Editing in a dialog

`mc_message_editor.js` is one native modal `<dialog>` shared by both surfaces,
and its contract is the whole of it: `open({title, text, placeholder, done,
cancel})`; `done(text)` may answer at once or with a promise, `{ok: false,
message}` keeps the dialog open with the message under the box, anything else
closes it; `cancel()` is told when the edit is given up, by Cancel, Escape, or
a second open replacing the first. Enter is Done, Shift+Enter a new line,
Escape Cancel (the `cancel` event taken over, and the key itself for a dialog
opened without `showModal`). Done is greyed out while an answer is on its way;
Cancel is not, because a save that takes a minute is not a reason to be stuck
in front of it, and a late answer to an edit that was given up is not shown.

Not full screen, unlike the system prompt editor. `fit()` sizes it from the
visual viewport -- the glass, which on a phone is narrower than the layout
viewport and shrinks for the keyboard -- no wider than 560, MARGIN from every
edge, a third of the spare room down so the keyboard, which takes the bottom,
leaves it alone, and never past the bottom margin whatever the measurement
said. The box grows with the words up to what the glass leaves once the
heading, the hint and the buttons have had theirs, then scrolls. It follows
`resize` and `scroll` on the visual viewport while open.

**The panel.** `startEdit` opens it and keeps only `editing` -- the
conversation, index, version and revision the Edit was pressed on -- and the
outline on the message (`markEditTarget`, now only on the conversation the
edit is in, because the thread on screen can change under an open dialog and
the same index there is somebody else's message). The composer is not part of
it: the draft stays, Send stays Send, the paperclip and microphone stay
enabled, `typed()`, `grow()`, `paste()`, `send()` and `render` have no editing
branch left, and the edit strip is gone from the DOM and the stylesheet. Done
is `saveEdit(text)`: the same envelope as before, aimed at the conversation
the edit began in, with `image_action: "keep"`; a refusal answers the dialog
with the reason and the status line with "Your edit is still in the editor."
A change of conversation under an open dialog no longer puts the edit away --
there is no shared box to fight over. Escape while the dialog is open is the
dialog's: `documentKey` returns before the precedence list, as it does for
the system editor, and the host's `assistantEditing` branch went with the
in-box edit.

**LLM Studio.** The edit row is still what the server knows, and Python still
decides everything. `llm_studio.js` watches the row (`offsetParent`, the
file's own test of showing, in a MutationObserver on `class` and `style`),
opens the dialog with the box's words a frame after the row appears -- the
row and its words arrive in the same update -- and finishes the press through
the row's own buttons, which now carry ids: Done writes the words into the
box, dispatches `input` (Gradio reads a box on `input`, not on assignment)
and presses Save; Cancel presses Cancel. A page without the dialog's script
has the row, as it always did.

**Traded away, knowingly.** The row's paperclip -- change or remove the
picture on a message being edited -- is behind the modal while the dialog is
up, and the dialog has no picture control: "a simple pop up ... current text
in an input field, cancel, and a done button" is what was asked for. A
message's picture is kept as it is by an edit from either surface now. The
README says so; if a picture control is ever wanted, it goes in the dialog,
not back in the row.

### The ⋯ menu, in groups

Recommended and taken up: four groups headed by what their entries act on,
in the order they are reached for, and the thread list one level down.

| Group | Entries |
| --- | --- |
| Chat · *character* | New chat, Threads ›, System prompt… |
| Composer | Auto Attach, Send to Generate |
| Panel | Free Float, Customize… |
| Models | the host's utilities (Unload All Models, Unload LLM) |

A heading is `groupLabel`: a `div` with `role="presentation"`, on the screen
for the eye and out of the menu's list of items for a screen reader, and the
Chat heading carries the character's name -- which is where the selector line
went. New chat is the old New thread under the asked-for word (the tab's
button keeps its own); its status lines say "chat". Threads is `threadsItem`,
`aria-haspopup="menu"`, and pressing it is `showMenu("threads")`: the same
box, redrawn with a **‹ Back** entry at its head, the character's threads
with the current one marked, the empty message ("No threads yet. Start one
with New chat.") and Cancel. There is no room beside a panel on a phone for a
menu that opens sideways, and a list that replaces the list says so with Back.

`toggleMenu` gained an owner. `menu.dataset.owner` is the button that opened
the menu and `menu.dataset.which` the list showing, so ⋯ closes its menu
whichever of its lists is up, `applyChat` closes the workspace menu by its
owner, and the stylesheet's checkbox column keys on `which`. `showMenu` is
where every list, submenus included, gets its Cancel; the test that held
`toggleMenu` to that now holds `showMenu`.

### The Literal Prompt boxes

No title above either box (`show_label=False`); the description is the
placeholder, in the quiet type the two prompt boxes above use, gone the moment
somebody types. The label is kept for the accessible name and the Extra
Networks browser's memory of which box was used last, and simply not drawn.
`LITERAL_POSITIVE_HINT` and `LITERAL_NEGATIVE_HINT` say which is which and
that Negative Literal is not Forge's Negative Prompt -- the one thing people
mistake it for, said where they would make the mistake.

### Verified

Node: `test_assistant_v5_js.py` (Chat, the shorter rows, the grouped menu, the
Threads submenu, the owner), `test_assistant_taps_js.py` (presses on the
header, the drag's tail, a press on a control), `test_message_editor_js.py`
(the contract, the keys, the glass arithmetic at four viewports, the look),
`test_assistant_v4_js.py` (the box's editing tests translated to the dialog),
`test_llm_studio_js.py` (the row watched, Done and Cancel through the row's
buttons, ids compared across the two languages). The tests that named the
Focus button's `aria-pressed` read the panel's `data-focus` now, and the ones
that named the accordion heading press Chat. Every new check was reverted
against and failed.

---

## 3.23 After the fifth round: New chat in sync, the menu inside the panel, and the picture cap

> "I tried to start a new thread from the ... menu, but the old thread stayed!
> ... I wanted [the menu] to stay within the boundary of our flyout ... simply
> replacing what would have been the window for the conversation and user
> prompt ... A new thread should make the flyout and conversation mode start
> from scratch because they are in sync."

### Why New chat looked like nothing

It worked, for about fifteen seconds. `createThread` sent `create_thread`, got
the new thread back and `select`ed it; the panel drew the greeting. Then the
next look -- §3.21's `check()`, which runs when the panel opens, the page
comes back, the workspace changes and every fifteen idle seconds -- read the
bootstrap's `selection`, found LLM Studio still on the old thread, and
followed it. §3.21 had written the look to follow the tab's selection whenever
it *differed* from the panel's, which was the feed's `character_changed`
reconstructed for polling, minus the one thing an event carries: that
something changed. So the panel was pulled back to a thread the tab had never
left.

Two fixes, because the report had two halves. `Store.noteTab` remembers where
the tab was last heard (the bootstrap's seed, the feed's `character_changed`,
every look) and the look follows only a *move* since then. And the tab now
moves with the panel, because "they are in sync": `_open_from_event` behind a
hidden box and button (`open-at`, `open-now`) opens the thread the panel
names -- the list refilled with it chosen, the thread state moved, the
transcript redrawn, the selection remembered and published -- and the panel's
`openInStudio` writes the id and presses it after New chat and after a choice
in the Threads submenu. Read-only like the refresh bridge beside it: nothing
is ever lifted out of a thread by a press from elsewhere, which is why it is
not `_open_thread`. (The refresh bridge itself, it turned out, is pressed by
nothing since the event spine went; it stays for a page that still has it.)

### The menu inside the panel

§3.14 fixed the menu to the window because the panel clips what it contains.
Asked for instead, in so many words: the conversation's place. `applyMenu`
hides the body -- or the workspace row, collapsed -- while a menu is up, the
menu is an ordinary child of the panel's column with `min-height: 0` and
`overflow-y: auto` so a long list scrolls inside the panel's height, and the
panel is placed again for its new size. Chat with a menu up puts the menu
away and shows the conversation whether or not it was showing before.
`placeMenu`, `MENU_GAP`, `MENU_EDGE` and `MENU_FLOOR` are gone, and so are
the menu's border, shadow and surface: it is the panel's inside.

### Taps on the header, under pointer capture

Reported once #229's build was on the host: "Triple tap to focus toggle is
not working for the flyout. It still works for the flyout button." §3.22's
`headerClick` listened for `click` on the header, and the node tests called
it directly. In a browser the drag takes pointer capture on the panel at
pointerdown (`startDrag`), and a captured pointer's click is dispatched to the
common ancestor of where the pointer went down and where it came up -- the
panel, once capture has moved the pointerup there. The header's listener
never fired. Now `startDrag` notes at pointerdown that the press began on the
header's space (`headerPress`; `supersede` clears it at every new press), the
listener is the panel's, and the click is read against the note wherever it
lands. The taps tests go through `supersede`, `startDrag`, `endDrag` and a
click on the panel, which is the shape that would have caught it. And asked
for with the report: two presses on the space go back to the previous
workspace, as two on the launcher do; `tapHeader` is `tapLauncher` without
the single press.

### The picture cap: 640

Asked as a recommendation, with the use named: a pose, an outfit on a
subject, never a face. Two families are in the catalogue and neither spends
tokens the way the 300-token estimate in `prompt.py` assumes. Qwen 3.5's
encoder takes one token per 32 by 32 pixels (patch 16, merge 2): 768² is 576
tokens, 640² is 400, 512² is 256. llama.cpp sizes a Gemma 4 picture to the
nearest multiple of 48 and spends a token per 48-pixel patch (its issue
#24146, closed as not planned in June 2026, is the record: the reference
processor fills a fixed budget instead, which llama.cpp does not): 768 is
16×16 = 256 tokens, 640 lands on 13×13 or 14×14 = 169 to 196, 512 on 11×11 =
121. So on this host, where the Intel card reads a prompt at a few tens of
tokens a second, 640 is a third of the picture's share of every pictured turn
saved for a loss nobody doing pose or outfit work will see, and 512 is half
saved for a loss they may: a full-length figure at 512 on the long side has
its garment a couple of hundred pixels wide, where a print, a buckle or a
piece of jewellery goes. `MAX_SIDE` and `VISION_MAX_SIDE` are 640.

### Verified

Node: the look's new tests (`test_assistant_recovery_js.py`: a look follows a
move and not a standing selection, a New chat survives the looks after it),
the in-panel menu (`test_assistant_js.py::TestTheMenuIsInsideThePanel`), the
bridge from the panel's side (`test_assistant_v5_js.py`) and from the tab's
(`test_llm_panels.py::TestTheTabFollowsThePanel`, including that it never
lifts a message and that the ids match across the two languages), the cap's
tests translated to 640. Every new check was reverted against and failed.

---

---

## 3.24 The focus root is a contract now: the WanGP tab fills the window

Mini Paint NEO's WanGP tab is a WanGP page in a frame, with the tab's own
*Integration management* accordion under it, sized by a script of its own to
the room left in the window. Under focus mode that gave a nearly full window of
WanGP with eight pixels of this rule's padding around it and the accordion on
the bottom edge, and the user asked for the frame to be the whole page — no
accordion, no reload.

Nothing changed here. That tab's stylesheet now keys a rule on the class
`enter()` puts on the focused panel, `forge-assistant-focus-root`: the panel's
padding and border go (an id-and-class selector with `!important`, over this
stylesheet's lone class with `!important`), the accordion is not displayed,
and that tab's sizer reads the same class to give the frame the panel's whole
height. So the name of that class is a contract with another repository from
today, the fourth thing shared with it after the dialog layer, the
`minipaint:overlay` event and the reading `visible()` makes of a hidden panel.
It is named in the README's focus passage for that reason: renaming it here
would break the WanGP tab's full page silently, since nothing on this side
would notice and that tab would simply keep its accordion and its padding.
The measurement lives on their side — `tests/browser_intercept.py` there
carries a copy of this stylesheet's focus-root rule (`FOCUS_STAND_IN_CSS`), so
a change to the rule here should be mirrored there too, or their check goes
on measuring a rule this extension no longer has.

## 3.25 Leaving the end of the transcript, and a character screen with no prose

### The transcript would not let go

"When I am looking at a live conversation, it is very hard to detach from the
bottom." The follow rule was the scroll position alone: at every scroll event,
following was *at the bottom within 100 px*. A wheel notch scrolls smoothly, so
its first events are inside that slack; a chunk of the reply arriving in that
instant rendered with `wasFollowing` true and put the reader back at the end,
and the next notch met the same thing. Only a scroll that cleared 100 px
between two chunks got away, and while a reply streams there is no such gap.

Leaving is read from the gesture now, before the scroll it starts has moved
anything: an upward wheel that has travelled `LEAVE_WHEEL_PX` (40 px, one
notch in either browser, more than a brush of a trackpad) within
`LEAVE_WINDOW_MS`, a finger dragged `LEAVE_TOUCH_PX` down the glass, or ↑,
Page Up or Home on the transcript — `leaveBottom()`. The position rule
(`scrolledTranscript`) only ever *re-follows* on a scroll that moved down and
reached the slack; an upward scroll still inside it changes nothing, and a
move this code made itself (`toBottom`, a render that kept the reader's place,
a revealed row scrolled into view) records its own position first so the
scroll event it fires does not read as the reader coming down. Leaving by the
scrollbar alone still takes more than the slack, on purpose: a drag of a few
pixels is not a decision. `tests/test_assistant_js.py::TestLeavingTheEnd` has
the exact case — one notch up, a chunk's render, the render's own scroll event
— and the wheel, touch and key gestures; seven mutations of the rule fail it.

Superseded by §3.32: the threshold, the window and the slack are gone, at
the user's request.

### The character screen

"Really bad right now": three system-prompt surfaces with a paragraph under
each, the sampling in an accordion whose relation to Save was explained in a
hint, the card import taking a third of the screen. The same components and
the same wiring, laid out in the order a character is written — picture, name
and greeting in one row; Context; **System prompt** as one labelled group
(*In force*, **Copy into override**, the override box); Voice; Save and
Cancel — with every `info=` and hint paragraph removed, and the sampling as
**Generation** and the import as **Import a character card**, two closed
accordions under the editor. The ⤢ System prompt editor is untouched. The
sampling stays outside the editor because it is in force while talking and
remembered on the character as it is moved (`_remember_sampling`); putting it
inside would have hidden a live control behind Edit.
`tests/test_system_prompt_editor.py::test_the_character_screen_is_terse_and_grouped`
holds the screen to that: no `info` under a box, the three accordions, no
prose.

## 3.26 What is free, in the header

Asked for from use: the grip's empty space shows free system RAM and each
NVIDIA card's free VRAM, grey and italic, "just numbers, really quick to read",
with no new live connection. `GET /memory` on the conversation prefix answers
`{ram: {free, total} | null, cards: [{index, name, full_name, free, total}]}`:
RAM from `mc_broker.free_ram_bytes()` (the OS's available figure), cards from
one nvidia-smi run (`detect_gpus`) rather than torch, because a torch reading
makes a CUDA context on every card it reads, WanGP's included. A reading
answers every page for a second, and the route is a plain `def` so the
subprocess runs in Starlette's threadpool, not on the event loop. The panel
asks once on opening and then every five seconds, timed from each answer, with
a four-second deadline; a closed panel and a hidden page ask for nothing, and an
answer arriving after either is dropped. A failed ask leaves the last figures,
dimmed (`forge-assistant-memory-stale`). The figures are `pointer-events: none`
inside the grip, so they drag like the rest of it and take part in the header's
two and three presses. An integrated Intel GPU has no VRAM of its own: its
memory is the RAM figure.

## 3.27 WanGP's Generate, in the header

Asked for from use: on the WanGP tab, a button beside the ⋯ that generates
"whatever is on the page", and adds it to the queue when something is already
generating, because WanGP's own buttons sit somewhere else on its page. It is
`window.minipaintWanGP.generate()`, Mini Paint NEO's (its
`docs/wangp/CONTRACTS.md`): a queue request with nothing overridden, so the
bridge writes WanGP's own generate trigger when idle and its add-to-queue
trigger when busy, and WanGP's own chain reads the live form. Not
`minipaintInterop.wangp.enqueue`: the outbox composes a job from WanGP's
*recorded* form (or the model's defaults, with inheritance off), which is not
what is on screen.

Shown only when the active workspace is `tab_wangp` and the bundle offers
`generate`; disabled until `state()` says ready with the queue. Looked at
again on every navigation, on opening, and on the free-memory reading's beat,
since nothing announces WanGP coming up. It never locks ("I should be able to
spam the button"): the bridge holds WanGP's live form for one request per page
and answers a second with `QUEUE_BUSY` until the first is confirmed, so presses
line up in the panel (`_generateLine`) and are sent one after another, each
answered in its turn, with the count still to go as the button's `data-count`.
The sender stops in the same turn it finds the line empty, so a press landing
just after starts the next one rather than waiting on one that ended. Each
answer (`started`, `queued`, or the refusal's own message) is a held status
line; once the line is empty the button's `data-state` and tooltip say how it
went for three seconds, `failed` if any press in it was refused.

## 3.28 Warm LoRA, Cold LoRA, in the ⋯ menu

Asked for from use, after a reading of the user's logs: "I want there to be a
toggle in the flyout menus '...' menu to turn this setting on and off. 'Warm
Lora' or 'Cold Lora'. Warm means stays on system ram, and cold means its dumped
from system ram and our extension manages the blob model, and evicts it on
change... never stack another lora on a blob model! The setting should
immediately free up the ram, and be capable of being turned on and off without
a image generation start. ... Once gone, turning warm back on does not load the
lora back!"

The setting is `mc_lora_ram` (the README's *Warm LoRA, Cold LoRA*), stored as
the Forge option **LoRA originals in system RAM** so the Settings page and the
menu are one switch. In the ⋯ menu it is two `menuitemradio` entries under the
**Models** group, **Warm LoRA** and **Cold LoRA**, the current one checked with
the same accent tick the switches use; the group is always drawn now, with the
host's unload entries after them. Each entry's tooltip says what the loaded
model holds: "12.1 GB of un-merged weights in system RAM now", "the loaded model
carries its LoRA baked in", or "no LoRA is merged into the loaded model".

The state is the server's and rides the header's free-memory reading (3.26):
`GET /memory` answers with a `lora` block — `{mode, merged, blob,
originals_bytes, freed_bytes, pending, model}` — read fresh on each reading
while the figures stay cached, so the menu needs no request of its own, and the
tooltip over the figures gains "LoRA originals 12.1 GB in system RAM (Warm
LoRA)" or "LoRA baked in (Cold LoRA)". A press on the entry that is not on
closes the menu and `POST`s `/lora-ram?mode=cold` (or `warm`) — a query
parameter, so the handler is a plain `def` in Starlette's threadpool: choosing
Cold frees gigabytes under the host's queue lock and the extension's model lock,
and never on the event loop. The answer's `message` is held in the status line
(`tell`): "Cold LoRA: 12.1 GB of un-merged weights freed from system RAM. The
loaded model keeps its LoRA baked in; a LoRA change reloads the checkpoint from
disk." A press while a generation runs is told that the originals are freed when
it ends, and they are. A refused press leaves the mode as the server last said.

## 3.29 The settings column in the panel, and a results column that fills the window

Two asks against Txt2Img, made with screenshots of the user's Forge (Lobe,
split previewer):

> "lets add this new option. A third state for the flyout menu, that removes the
> left column from the text to image tab, and makes it scrollable in the flyout
> menu. Add this button to the flyout menu bar only when text to image tab is
> open."

> "the right column (generate, gallery, gallery buttons) does not scale all the
> way to fill the space. There is a gap at the bottom ... The right column
> should not scroll, it should just fill the space."

Both live in `javascript/forge_assistant_layout.js` (`Dock`, `Fill`), with the
panel's side in `forge_assistant.js` (`applyChat`, `dockShown`, `syncDock`,
`toggleDock`) and the rules in `style.css` beside focus mode's.

### The column is drawn in the panel, not moved into it

The obvious build -- take `#txt2img_settings` out of its row and append it to
the panel -- breaks Forge. The panel is under `<body>`; Forge's scripts find
the prompt boxes through `gradioApp()`, the `<gradio-app>` element (token
counters, Ctrl+Enter, extra networks writing into the prompt, paste,
`get_tab_index`), and so do this extension's own scripts and other
extensions'. A column outside `<gradio-app>` is a column none of them can find.

So nothing is moved. The column gets `position: fixed` and the pixels of an
empty placeholder in the panel (`.forge-assistant-dock`), written inline with
`!important` and re-written after every placement, drag and resize of the panel
(`syncDock`, from `placeNow` and `moveDrag`). Its row gets a class that hands
its one remaining track to the results column and hides Forge's resize handle;
Forge rewrites the row's grid template inline on every window resize, without
priority, so the class wins without anything having to undo Forge. `release()`
puts back exactly the inline values it replaced, priority included.

The column is drawn *under* the panel, and the panel is see-through over the
placeholder and lets presses through it. The other way round cannot work in
focus mode: the focus root is a stacking context at layer 1100, below the
panel's 1200, and nothing inside it can be drawn above the panel whatever its
own z-index says. Under the panel works in both. Two traps met on the way,
both now held by `tests/test_assistant_layout.py`:

- `#forge-assistant-root > * { pointer-events: auto }` is an id rule; the
  docked panel's `pointer-events: none` needed the id too, or the panel took
  every press meant for the column.
- A Gradio column given a height wraps instead of scrolling (`flex-wrap: wrap`
  on every column, Mini Paint NEO's trap from its Clipboard): the docked column
  pins `nowrap`, and its children `flex-shrink: 0`, because Lobe makes the
  column a flex column with `!important` and the blocks would otherwise be
  squeezed to fit rather than scrolled.

A transform, filter or containment on an ancestor makes `position: fixed`
resolve against that ancestor (focus mode checks for the same trap). `place()`
reads where the box landed and takes the difference back once.

The docked panel was first as tall as the window. Asked for instead: "maintain
the same height restriction as the fly out menu in conversation mode. It should
not go to the top and bottom of the page ... the entire columns worth of
content should be available still. Just need to scroll it." So the placeholder
is given the conversation body's height with its transcript at its stylesheet
maximum (`sizeDock`): the transcript's own `max-height`, read from the
stylesheet, plus everything else in the body as measured. The body is hidden
while docked, so it is shown for the measurement and hidden again in the same
turn, and nothing is painted in between. The placeholder may shrink
(`flex: 0 1 auto`) when the panel is capped to a phone's visible height.

The state is the panel's (`settingsDocked`, saved with the rest of its layout
for the tab). It shows only on Txt2Img; elsewhere the panel shows the
conversation or the tab bar as `conversationExpanded` says. (Since §3.41
Img2Img has the same state, its own: `img2imgDocked`.)

While it is on, the column stays out of the page: `syncDock` places it over the
placeholder when the panel is open in the docked view, and otherwise keeps it
docked and hidden (`visibility: hidden`, still out of the row, so the gallery
keeps the whole width). Only switching the panel to the conversation (Chat) or
the tab bar (🎛 again) gives it back. The first build gave it back whenever the
panel closed, and was reported against: "it returns to the left column when i
exit focus mode. if i exit focus mode with the column enabled inside the flyout
menu, it should remain hidden when focus mode exit. the only way to get it back
is open the fly out and switch to conversation or tab mode". Focus mode never
touched the column; the panel closed in focus mode, for the whole gallery, had
already put it back, and leaving focus showed it.

### Forge's mobile script, and a docked row one track wide

Reported after #260, with the Lobe theme's split previewer on: "Enter focus
mode, enable generation in flyout (now full screen gallery), exit focus mode
with enabled generation in flyout, disable generation in flyout, enter focus
mode, see generation is now in the left column" -- Generate's box back in the
toprow's actions column, where Gradio first drew it, instead of the gallery's
container where Lobe keeps it. No script on the page was known to move it
there, and a real Gradio 4.40 page with Lobe's own bundle, Forge's `script.js`
and resize handle, this extension and real presses (the browser granting full
screen) did not reproduce it in Chromium through every sequence of focus, dock
and panel toggles up to four steps. So a temporary watcher went onto the
user's page (`model_chain_watch_generate.js`, since removed): every DOM call
that moved the box, with the caller's stack, into `model_chain.log`.

The log named **Forge Neo's own `extensions-builtin/mobile/javascript/mobile.js`**.
On every window `resize` it asks whether the page is a phone -- a results
column that is laid out with `offsetLeft === 0`, which is where a stacked
layout puts it -- and when the answer changes it moves `#txt2img_generate_box`:
into `#txt2img_results` for a phone, back into `#txt2img_actions_column` for a
desktop, caring nothing for where a theme had put it in between. The docked
row is one track wide (`.forge-assistant-docked-row { grid-template-columns:
minmax(0, 1fr) }`), so the results column sat on the row's left edge: a phone,
to that script. And the browser's full screen, which comes and goes with
focus, resizes the window. Exiting focus with the column docked: resize,
"phone", the box into the results column. Undocking, then entering focus:
resize, "desktop", the box into the actions column -- in the left column,
inside the toprow that Lobe had moved into the settings.

So the docked results column keeps a pixel off the edge. `Dock.place` reads
the results column's `offsetLeft` *before* the row changes (`besideOf`): a
column drawn beside the settings gets `forge-assistant-docked-beside`, a
`margin-left: 1px !important`, and never reads as the edge while docked; one
drawn under them -- a phone, at the edge already -- gets nothing, so the dock
turns no phone into a desktop for that script (which would have moved Generate
out of the results column into the toprow in the panel). `release` takes the
class off. The pixel comes off the gallery's width, which nothing can see.

`tests/test_assistant_layout.py` carries that script as it is in Forge Neo and
Lobe's shape of the toprow inside the settings, and holds both halves: docked
beside the settings, two resizes around an undock leave the box in the
gallery's container (without the margin the first resize moves it); stacked
under them, the dock leaves `offsetLeft` at 0 and the box where the script put
it. Checked only there and against the user's log; the box the script had
already moved on the user's page comes back to the gallery's container on a
reload, when Lobe places it again.

### The fill is measured, not a share of the window

The user's user.css set `#txt2img_gallery_container { height: 85vh !important }`.
Outside focus that lands near the bottom by luck of the header's height; in
focus the header is gone, the column starts higher, and 85vh stops short --
the gap in the screenshot. No single fraction of the window is right in both,
so `Fill.measure` reads it: from where the column rests (its place in the row
with the scroller at its start, or where it is held, if sticky and held lower)
to the bottom of whatever scrolls it, less the column's own content above and
below the gallery, is the gallery's height. One custom property on the results
column carries it; the stylesheet applies it on two ids with `!important`, so
the user's one-id rule loses whichever sheet loads last and the user's file
needs no change. Measured again on window and visual-viewport resizes, on
focus entering and leaving, and whenever the results column changes size
(a ResizeObserver: a tab shown, the dock, an infotext arriving). It writes only
when the value changed, so its answer to its own write finds nothing to do.

Where the column is held by `position: sticky` is read from the layout, not
computed from `top`: Chromium measures a sticky offset from inside the
scroller's padding, which put the first version 8 px past the window's bottom
in focus mode, and Firefox (the user's browser) need not agree with it. A
sticky column drawn lower than its place in the row is being held, and where it
is drawn is where it is held.

`forgeAssistant.fill.explain()` in the console says, per tab, what was measured
or why the column was left alone ("stacked under the settings" on a phone).

### What is kept in view, and what goes below it

The first build fitted the *whole* column into the window: everything in it
that was not the gallery was taken off the gallery's height. Under Lobe the
infotext is a tall table (a Positive Prompt block and a row per setting), so
after every generation the gallery was about half a window, under it. Asked
for instead: "The gallery need to fill its area, only bounded by the screen
border, generate / interrupt / cancel, progress bar (when in progress), and
gallery buttons. Everything else should be off page. The image should scale up
to fill the maximum allowed space. The padding above generate and below gallery
buttons should be the same, giving view of gallery in perfect center ... All
other content below should still be rendered, but off page and below the view,
thus requiring a scroll to see it."

So `keptBox` takes the part of the column that stays in view: from the first of
Generate's box (Lobe's split previewer moves it into the gallery's container;
Interrupt and Skip share it), the progress bar while one is in the column's
flow (Forge puts `.progressDiv` just before the gallery's container, and Lobe
makes it `position: relative`; Forge's own absolute bar is an overlay and is
not counted), the container and the gallery, down to the gallery's buttons.
The gap from the top of the view to the first of those is left again under the
buttons, and the gallery is what remains. A second custom property puts room
under the buttons so that what follows them -- the infotext group -- starts
exactly at the view's bottom: worked out from the height as it will be drawn,
rounded down to a whole pixel, because a theme's fractional sizes otherwise
left the infotext's top edge a fraction of a pixel inside the window.

The top of the view is the scroller's own top edge when something other than
the page scrolls the column (focus mode's root). When the page scrolls,
`coveredTop` asks the page what is drawn at the top of the window above the
column (`elementsFromPoint`), and the bottom of a sticky or fixed element there
-- Lobe's header -- is the top of the view; this extension's own panel is never
that element, neither is anything holding the column, and neither is anything
fixed over more than half the window (a lightbox, a dialog: a header is a bar).
A second bar under the first is looked for too.

That question has one answer the page cannot give: during Forge's divider drag
(`body.resizing`), when its stylesheet gives every element `pointer-events:
none` and a hit-test finds nothing. Reported as "When i move the column divider
to adjust column widths, sometimes the generation button appears to fall under
the tab bar": a measure during the drag took the window's top for the ceiling
-- Lobe keeps the tab buttons inside its header, so the DOM walk above the
column (`aboveBottom`) found nothing either -- and the panel sat under the
header until something else resized the column. Under the drag the fill now
keeps what it wrote (`dragging`, `deferred`), and the body's class coming off
is the cue to measure again.

### Verified

In Chromium, against a Forge-shaped Gradio 4.40 app (the real markup, Lobe's
rules, the user's user.css) while building, and in the suite by
`tests/test_assistant_layout.py` (a page with only what decides the geometry,
one fractional height included) and `tests/test_assistant_dock_js.py` (the
panel's states under node). Every
rule and branch above was reverted one at a time and a test failed for each,
except two that change nothing on that page: the focus hook (the column's own
resize already asks for a measure) and the `!important` on the gallery's height
(a two-id rule outranks the user.css without it). Neither has run on the
user's machine or in Firefox.

### The theme's one scroller, and the dock (2026-10-17)

The Model Chain theme orders the page's scrolling (docs/24 §0, "One scroller
for the page"): the settings column first, the results column held where it
rests, then the results with the settings held, by sticky positioning from
two measured numbers on the row. The dock is its first stand-down: a docked
column is `position: fixed` by the rule above, out of the row, and the theme
writes nothing for it, so the row is the results' height, nothing holds, and
the page scrolls the results alone while the column scrolls in the panel --
whose `overscroll-behavior: contain` keeps a wheel at the column's end from
moving the page. `release()` puts the column back and the theme measures
again (it watches the row and both columns for their size). The fill is
untouched: it reads where the results column is held off the layout, as
before, and the theme reads the same thing the same way.

## 3.30 Free Float on a phone

Reported against the build above: "on mobile the fly out menu will be forced to
a certain location ... make sure that when I'm in mobile the free float mode
works perfectly", and then: "the fly-out menu is docking instead of being able
to move anywhere in free float". Below 640 px the open panel was always the
sheet -- full width, against the top or bottom half -- whatever Free Float
said, so a finger's drag moved it and the release put it back.

Now the sheet is the anchored mode's alone: with Free Float on, `placePanel`
floats the panel at any width, non-modal, as on a desktop. Three more things
were needed for it to behave on a phone, each found by doing it in Chromium
with real touch input (DevTools' `Input.dispatchTouchEvent`, a phone's
viewport, `is_mobile`):

- **Where you let go is where it stays.** The fraction was taken of the whole
  window and laid out across the travel inside the gap, so every drop moved by
  up to the gap (6 px on a phone) and turning Free Float on moved the panel up
  to 23 px on a desktop -- the test of that only checked the fraction.
  `fractionOf` is now the exact inverse of `floatPoint` (`floatTravel`), and an
  axis with no travel -- a phone, where the panel is as wide as it may be --
  keeps the fraction it had for a wider window.
- **The header keeps its place.** A share of the room around the *whole*
  panel moved the header whenever the panel changed height: docking the
  settings moved it 75 px up on a phone, the 🎛 just pressed with it. Up and
  down the travel is the header's now (`floatHold`): the panel grows and
  shrinks below its header and moves up only as far as it must to stay on
  screen. A panel let go with its bottom past the window's is remembered as
  against the bottom (1) and stays against it at any height.
- **Never taller than what shows.** The stylesheet's cap is `100vh`, which on a
  phone is the screen with no keyboard and no address bar. The keyboard shrinks
  only the visible viewport, so the panel ran under it, composer and all.
  `placePanel` writes the cap from the visible viewport on every placement
  (the keyboard, the bar and a rotation all re-place). A test asks for pinch
  zoom to get the same shrink, because a browser will not show a keyboard.

Checked and not changed: Chromium does not report a finger's implicit capture
moving from the grip to the panel as a lost capture, so the drag's
`lostpointercapture` listener does not end touch drags. And a tap straight
after a flick is kept by the browser to stop the flick; the touch tests move a
frame at a time and rest before lifting, as a hand does.

## 3.31 A docked field takes a drag only once a tap has engaged it

Asked for once the column was in the panel: "when i try to scroll i might
scroll a text box, or move a slider. I would like scroll input fields or
sliding sliders etc, it should require a selection tap. So by default, the
entire content column is scrollable in the flyout menu, but once i want to say
type, i have to tap the input field first to select it, then i am in it, and i
can type. Same with slider ... I would like the component to throw some sort
of highlight state outline so i know i'm in".

`Guard` in `javascript/forge_assistant_layout.js`, attached by `Dock.place`
when the column docks and detached by `Dock.release`. While attached the column
carries `forge-assistant-guarded`, and the stylesheet gives every control that
takes a drag or a wheel for itself -- a text box, a number box, a slider, the
compact spatial canvas (`GUARDED`; the stylesheet's list names the same) -- no
presses at all. That is the whole of the scrolling half: the hit test passes
over a control with `pointer-events: none`, so a finger or a wheel that starts
on one is on the block around it, and the column is what scrolls. Nothing
listens for a drag and nothing calls `preventDefault` on one.

A tap is a `click` -- a browser fires none after a scroll -- caught on the
document in the capture phase. On a block whose guarded controls are all its
own (a slider's label, track and number are one block) it *engages* the block:
`forge-assistant-engaged`, a 2 px outline in `--color-accent` (an outline,
so nothing in the column moves), and its controls take presses again. The
engaging tap does nothing else (`preventDefault`: a label's own focus is
decided here): a text box tapped -- or a block with nothing to slide -- is
focused, caret at its end, so the keyboard comes up; a slider's track is
focused without a keyboard; its number, tapped, gets one. In a block that
holds several fields (an accordion) only the field under the point is meant,
and a tap on its header engages nothing and opens it as ever. A tap anywhere
outside the engaged block, Escape, the column going out of sight or undocking
lets go, and blurs what was focused in it. A control reached by Tab is engaged
too (`focusin`), outline and all.

Dropdowns (`.gradio-dropdown`), checkboxes, radios and buttons are not guarded:
a drag across them scrolls already, and they act only on a tap. Gradio's
dropdown is a text input, which is why the list excludes `.gradio-dropdown *`.

Then, asked for: "input fields should not scroll. They should always present
full content ... I dont want the view to scroll up to the top when i select it
to begin typing, i want what is in view to stay in view, so i can put the
cursor exactly where i need it", and "input field should never close on its
own in the flyout. Select should never auto deselect ... the input field will
deselect sometimes while i am still typing!"

- **Full content.** `Guard.fit` makes every docked text box as tall as its
  text (measured at `height: auto`, written in pixels with priority), with the
  stylesheet taking away its `max-height`, overflow and resize handle. It runs
  on attach, after every `input` (bubbling, so after Gradio's own resize in
  the same task), when anything in the column changes (a MutationObserver,
  its own writes taken back with `takeRecords`) and when the panel's width
  changes; undocking puts back the heights it found.
- **The caret where the finger was.** The engaging tap focuses with
  `preventScroll`, puts the caret at the character under the tap and puts the
  column's scroll back if anything moved it. The character is read off a
  see-through copy of the box's text laid over it (`caretAt`), because
  Chromium's `caretPositionFromPoint` over a multi-line textarea answers the
  first line whatever the point.
- **Only the reader lets go.** The guard released on every `click` outside
  the engaged block, and scripts on this page press hidden buttons with
  `.click()` -- each was a tap outside, letting go and blurring mid-word. It
  reads trusted clicks only now. And `Dock.place` released when the slot had
  no height for a moment, which a phone's keyboard causes (it shrinks the
  visible window, the panel is capped, the slot gives way first); it no longer
  does.

Checked in Chromium by `tests/test_assistant_layout.py`, with a mouse (the hit
test, a drag across an unengaged slider that moves nothing, the wheel over a
text box scrolling the column, the box showing all its text and growing with
new lines, engaging with the caret on the tapped line and the column not
moving, a script's click and an empty placement leaving it engaged, sliding,
typing, letting go) and with real touch on a phone's viewport (a drag on a text box scrolls the
column; a tap engages and focuses it). Taking the stylesheet's rule out, the
guard's attach, or its focus each fails both. Not run on the user's machine or
in Firefox.

## 3.32 Docked to the end of the transcript, with no slack

Reported again after §3.25: "When streaming response and my scroll is docked
to bottom, i try to use my mouse scroll to scroll away, but im still stuck!",
with the rule asked for: "The moment i scroll away from the bottom of the view,
i should be undocked from the bottom. If the stream continues, it will continue
down and out of view ... If i scroll to the bottom again, and reach the end of
scroll, then i become docked again" -- in the panel and in LLM Studio.

What §3.25 left: leaving wanted 40 px of wheel within 400 ms, and the end was
anything within 100 px of it, so a reader who had moved up less than that was
"at the end" and every chunk put them back. LLM Studio was worse: its rule was
the position alone, with the same 100 px, and Gradio's Chatbot autoscroll
re-pinned within 100 px of its own.

The rule now, the same in both (`scrolledTranscript`, `wheelTranscript`,
`touchTranscript`, `reachBottom` here; `watch`, `scrolled`, `wheeled` in
`javascript/llm_studio.js`):

- **Undocked** by any wheel upward (when there is anything above), a finger
  moved `LEAVE_TOUCH_PX` (4 px, more than a tap's tremble) down the glass,
  ↑, Page Up or Home, or a scroll upward that leaves the end (the scrollbar).
  Gestures are read as they start, before the scroll they cause.
- **Docked** by a scroll downward that reaches the end -- within
  `FOLLOW_END_PX` (2 px, for a scaled page's fractions) -- or by a wheel or a
  finger pushing down while already there, where no scroll event can come.
- A move the code makes itself writes its own position first, so its scroll
  event is never read as the reader's.
- LLM Studio's Chatbot is built with `autoscroll=False`.

Held by `tests/test_assistant_js.py::TestLeavingTheEnd` and
`tests/test_llm_studio_js.py::TestAnchoringTheTranscript`; the old "inside the
slack still follows" tests were rewritten to the new rule.

## 3.33 One space around the gallery, and a grid of whole rows

Reported against §3.29 with three full-screen screenshots (focus with the
panel, the page under Lobe, the grid view): "There is empty space above
generate and below the gallery buttons. I want that padding removed", then,
asked whether flush to the edges was meant: "It should not be flush ... The
amount of space between the bottom of the gallery to the top of the gallery
buttons should be the minimum padding we use for spacing. Make the space above
the generation button, between generate button and gallery, between gallery
and gallery buttons, and gallery buttons of the view." And of the grid view:
"There is empty space in the gallery, enough to see the semi checkered
background. Can you fix the sizing so that view which scrolls images fills the
gallery window with no cut off?"

§3.29 measured the gap from the top of the view to Generate and left it again
under the buttons. That gap was the theme's: Lobe's sticky offset under its
header, the results panel's padding, focus mode's own padding. Now one space is
used everywhere, and it is the theme's own gap between the gallery and its
buttons, measured each time:

- **Above the frame around the first thing kept in view.** The frame is the
  outermost box inside the column that the first kept thing is the first thing
  in (`frameOf`): the theme's panel around the whole right side -- Lobe's gray
  box, Forge's own `variant="panel"` -- whose padding is above Generate. Its
  top is brought to one space under the top of the view, or under whatever is
  laid out above the column and across it (`aboveBottom`: the tab buttons over
  a column the page scrolls; sticky and fixed things are `coveredTop`'s), up or
  down, with an inline `margin-top` on the frame, so the column itself, and
  where a sticky column is held, do not change; Generate sits inside the frame
  at its padding. The first build moved the first kept thing itself (the
  gallery's container, with Generate inside it) and was reported against at
  once: "The generate button is being rendered outside of that dark gray box
  ... I would prefer the generate button still render inside the gray box so
  that all things in the right appear within its boundary" -- a negative
  margin on a child pulls it up out of its parent's padding and past its edge.
- **Between each kept thing and the next** (Generate, the progress bar, the
  gallery), the same way -- each moved by its own frame, the outermost box it
  is the first thing in short of what it shares with the thing before it
  (`frameOf` with `commonAncestor`). Gradio wraps a group's components in a
  box that clips (`overflow: hidden` on the group and on its `.styler`), and
  Lobe's layout leaves a wider gap between Generate and the gallery than the
  space, so the first build's margin on the gallery itself pulled its top edge
  up out through that clip: reported as "the light gray stroke that appears
  around the image gallery ... appears on the left, bottom, and right... But
  not the top. In fact, depending how I size the column, a little bit of the
  top edge of the image gets cut off." The wrapper moves instead, and the
  gallery stays whole inside it. The gallery-to-buttons gap is the space by
  definition.
- **Under the buttons**, to the bottom of the view; the gallery takes the rest.

The margins are written inline with `!important` and the value each node had is
kept (`Fill.spaced`): every measure reads the column with them taken off and put
back in the same turn (`bare`), so it always reads the theme's own spacing and
never its own; the fill coming off puts them back. Moving is not resizing, so a
column that only moves (the tab buttons hidden, nothing else changed) is
measured again at the next resize, as before.

**The grid view.** The grid's scroller (`.grid-wrap`, found anywhere in the
gallery) is given the measured pixels down to the gallery's bottom edge, not a
share of a parent that may have no height, so no theme's cap leaves the
gallery's background showing under it. Its rows are given one height
(`--forge-assistant-fill-row`, with `data-forge-assistant-rows`): the whole
number of rows nearest the pictures' own shape at the grid's column width (the
first thumbnail's picture, read when it loads), stretched or shrunk to fill the
scroller exactly. Each thumbnail fills its row, and the scroller snaps a row at
a time (`scroll-snap-type: y mandatory`, `scroll-padding-top` at the scroller's
padding), so no row is cut by the gallery's edge at rest. A picture whose shape
differs from the row's is fitted inside it as the gallery's `object-fit` says.

Checked in Chromium by `tests/test_assistant_layout.py`: at rest, in focus, with
a progress bar, with an infotext, resized, with the theme's padding over the
column (brought up under the tab buttons, then under the header), the panel one
space under the ceiling with Generate at its padding edge, and the grid of
thirty portrait pictures inside an extra wrapper under a theme's `max-height`
cap. And by `tests/test_theme_layout.py` on the Model Chain theme's page with
these scripts loaded before the theme's, as Forge loads them: that theme's
header is sticky *inside the page's padding*, so at scroll 0 the first pixel
of the window is the page's own margin and `coveredTop` found no bar there and
put Generate under it; it now asks the rows just under the top too, eight
pixels at a time down to sixty-four, until something answers. The same page
showed what a theme that lets the gallery stop short of a tall container does
to the measure: the gap between the gallery and its buttons, read as the one
space, was half a window, and nothing fitted. The theme now grows the gallery
into its container, as Lobe's did. Each of these was reverted one at a time and a test failed:
the lift, the pair spacing, the ceiling, the bare read, the grid's pixels, the
row height, the snap and the refresh on a picture's load. Not run on the user's
machine or in Firefox.

### The fill and a zoomed-out page (2026-10-17)

The fill measures the visible viewport (`viewport()`, the glass: a phone's
keyboard shrinks it, and the gallery stays above the keyboard's edge).
Zoomed out -- a phone's browser lets a page wider than the window go below
1, and the Model Chain theme's hidden infotext paragraph had made the page
that wide (docs/24 §0, "The user's second run") -- the visible viewport is
taller than the window in CSS pixels, and the fill grew the gallery to the
zoomed-out glass, which was a gallery taller than the screen once the zoom
came back, with the theme's row grown to match. It takes the visible
viewport's height, and never more than the window's
(`documentElement.clientHeight`);
`test_zoomed_out_the_fill_is_no_taller_than_the_window` zooms a wide page
out through CDP in the mobile viewport and expects the same gallery.

## 3.34 A drag takes only the handle

Reported from the user's machine: "Sometimes when I drag it, it drags the
whole page instead of just dragging the fly out menu button or the fly out
menu itself. Can you make sure that the fly out menu when dragging the drag
action does not impact anything else on the page? If I drag the fly out menu
or its button, only that should move."

### What the browser does with a press nobody cancelled

`startDrag` took pointer capture and moved the panel, and left the press's
own default to the browser. On a desktop that default is text selection: a
press on the header -- its text, or the empty grip beside it -- began a
selection that grew across the page's text as the hand moved, and the page
lit up under the drag. Worse, a press *inside* a selection that already held
the handle (the last drag's, or a select-all) is a native drag of that
selection: the browser carries a ghost of everything selected -- the whole
page -- under the cursor, fires `pointercancel`, and the panel goes nowhere.
That is the report, and "sometimes" is the selection a previous drag left.
On a phone the default is the pan: the handles have said `touch-action:
none` since the first build, which is what tells the browser before the
gesture begins that a finger on them is not a scroll, but a host's stylesheet
can take it away (a reset's `button { touch-action: manipulation }` is
common), and then the page scrolled under the finger.

### Three things, each on its own

1. **A mouse or pen press is the shell's.** `startDrag` cancels the
   pointerdown once it is past its guards (a second button, a second finger,
   a press on a control are still the browser's). A cancelled pointerdown
   cancels the compatibility mouse events and with them the selection and
   the native drag, and leaves the click the press becomes alone: a click on
   the launcher still opens the panel, and the header's three presses still
   toggle focus. **A finger's press is not cancelled.** A finger selects
   nothing by dragging, and Chromium delivers the click after a cancelled
   touch pointerdown with `detail` 0, which `headerClick` reads as a
   keyboard activation: the first build cancelled every press, and three
   taps on the header stopped toggling focus on the phone -- found by
   driving the taps against HEAD's script and the build in Chromium, not by
   a test, which is why there is one now.
2. **The handles are nothing the browser may select or drag.** The launcher
   and the header are `user-select: none` and `-webkit-user-drag: none`, so a
   selection cannot start or end on them whichever way it is made, and a
   `dragstart` on either is refused where it begins.
3. **A finger's move under a live drag is cancelled** (`blockPan`): a
   non-passive `touchmove` listener on each handle -- touch events follow
   the touch that began on the handle wherever the finger goes, so the
   handle hears every move -- that cancels the move while `this.drag` is
   set and the event is still cancelable. It is wired on the handles and
   never on the window, on purpose: a non-passive touchmove listener on the
   window makes every scroll of the page wait for the main thread, which a
   Gradio page can hold for a long moment; on the handles it costs only the
   touches that start there, which must wait anyway. A press `startDrag`
   refused leaves the gesture to the browser, as before.

### The account, after a second report

The build above went to the user's machine and came back: "unfortunately
that didn't work, dragging the flyout button or menu still moves the whole
page" -- "i only touched the flyout and nothing else. just open the page and
tried to drag it once the page finished loading". A page just loaded has no
selection, so the mechanism above is not what they are seeing, and nothing
in Chromium here -- a mouse, a finger, a select-all, a host rule on the
handles -- moves anything but the handle. What their browser does with the
press is not known from here. So every drag keeps an account of itself
(`watchDrag` at the press, `accountForDrag` at the end): the pointer and
what it pressed, the pressed handle's computed `touch-action` and
`user-select`, whether the press was cancelled and captured, the page's
scroll, the visual viewport's offset and scale, the document's height, the
tabs' place in the document and the selection's length before and after,
the moves seen and the furthest one, the touch moves the handle heard, how
many were cancelable and how many it cancelled, how the drag ended
(`pointerup`, `pointercancel`, `lostpointercapture`, or a reset) and whether
the document was full screen. It is one `Forge Assistant drag:` line on the
console with the account as JSON, kept as `forgeAssistant.shell.lastDrag`;
and a drag that moved and found something else moved -- the page scrolled,
the viewport moved, the tabs moved, the height changed, a selection grew,
the browser ended it -- says so on the status line (`tell`, a warning), where
a phone can read it with the panel open.

Asked for that console line, the user answered: "i meant our log file ...
`model_chain.log` ... write it here, with other logging content". So the
account goes to the server -- `POST /model-chain/conversation/v2/journal`,
behind the conversation API's key like every other route -- and
`mc_assistant_journal` writes it as one line of `model_chain.log` through the
`model_chain` logger (`mc_logfile` writes that logger to the file, and the
console gets it too): the summary first, `Forge Assistant drag (mouse on the
launcher, 180 px, ended by pointerup, the page scrolled by 0, 240)`, then the
account as JSON, so a drag sits between the lines of whatever else was going
on. For that the account gained the events as they came (`trace`: the first
sixty and the last, each with its time, what it was, where it was and what the
browser said of it -- cancelable, cancelled -- the rest counted as
`dropped`), where the handle was put (`put`, `from`, `box`), and the page's
environment: the browser, the pointer the device has (`maxTouchPoints`,
`pointer: coarse`, `hover`), the pixel ratio, the window, the visual viewport
and its scale, the screen, the document's size against the window's, the
scrolling element, what the root and the body say about overflow, transforms,
position and zoom, the theme, focus mode, full screen, free float, the panel
open or a sheet. Nothing of it is read into anything on the server, on
purpose: the point is what the browser said. The console line and the status
line stay; a server that refuses the post or is not there is a console line
and never in the way of the drag. The next report is that log file.

### The log's answer, and the hold

The user sent `model_chain.log`. Eleven drags, 16:48:52 to 16:49:24, every
one `pointer: touch`, on a browser whose account reads: `Mozilla/5.0
(Windows NT 10.0; Win64; x64; rv:153.0) Gecko/20100101 Firefox/153.0`,
`touchPoints: 0`, `coarse: false`, `hover: true`, a 886 × 898 window on a
886 × 983 screen (taller than wide) at a pixel ratio of 2.22, the theme on,
the document 886 wide (no sideways overflow) and 1620 tall once the page
had filled in. A Gecko browser taking a finger while calling itself desktop
Firefox with no touch points: LibreWolf on a Windows touchscreen under its
fingerprinting defence, which spoofs the touch points and the pointer
queries, or Firefox for Android with *Desktop site* on, which spoofs the
same; the user has not said which.

What every drag that moved says: `touchAction: "none"` and `userSelect:
"none"` computed on the pressed handle, `pressCancelled: false` (a finger,
by design), `captured: true`, every touchmove the handle heard **cancelable
and cancelled** (`touchMoves` = `touchCancelable` = `touchBlocked`: 52, 77,
107, 44, 34, 39, 27), the drag **ended by its own `pointerup`**, never a
`pointercancel` -- and `pageScrolled` 282, 67, 473, −448, 484, −398: **the
page panned by the finger's travel, against it** (the finger up, the
content up), while the handle moved too (`put` against `box`). The first
drag, on a page still 898 tall, scrolled nothing because there was nothing
to scroll. Nothing on the main thread scrolls the page during a drag (no
script here does, and the browser's own scroll anchoring excludes fixed
boxes), so the pan is the browser's compositor, which in Gecko pans on its
own once the page has not answered a touch within its content-response
window (a few hundred milliseconds) -- a Gradio page can be busy longer --
and which, on the evidence, did not read the handle's `touch-action` either.
Whatever the page says about the gesture, in a listener or a rule, arrives
after the decision.

**So the page is made unscrollable for as long as a finger drags** (`lockScroll`,
`unlockScroll`): `overflow: hidden` on the scrolling element, with
`scrollbar-gutter: stable` so a classic scrollbar's width does not shift the
page under the handle, and under focus mode on the focused root as well,
which scrolls in the page's place; inline, and put back as found by
`endDrag` on every path a drag ends by (its pointerup, a cancel, the shell
disposed). A compositor cannot pan a frame that is not scrollable, and the
hold is painted as soon as the main thread is free, which is the same moment
the press is handled -- so a page that was busy at the touch may still move
a little before the hold lands, and no further. A mouse never pans, so a
mouse drag holds nothing. The account gained `delay` -- how long the press
waited for the page, the event's stamp against the handler's clock, which is
the number that would confirm the busy-page reading -- `locked` (what the
hold took), and the page's scroll sampled along the trace (`sy`, ten times
a second: when it starts to move against the finger is the question).

### Verified

`tests/test_assistant_js.py` (`TestADragTakesOnlyTheHandle`): an accepted
mouse press is cancelled and a pen's too; a finger's begins the drag
uncancelled; a second button, a second finger and a press on a control are
not cancelled and begin nothing; a move is cancelled only under a live drag,
never one the browser already took, and not after the drag ends; the blocker
is on the handles and no touch listener is on the window.
`tests/test_assistant_layout.py` in Chromium: a mouse drag from the grip
across the page's text moves the panel by exactly the drag and selects
nothing; with the whole page selected first the panel still moves; a
`dragstart` on either handle is refused; on a phone a finger takes the
launcher 200 px up a page three windows tall and `scrollY` stays 0 while a
finger on the page's text does scroll it; with a host rule `touch-action:
auto !important` on both handles the page still stays still under the drag;
and three taps on the header still toggle focus. The account: a quiet drag's
line says the press was cancelled and captured, the handle's rules, nothing
moved; a page scrolled under a drag (by the test, mid-drag) is in the line
and on the status line; a finger's line counts the moves the handle heard
and blocked. The post (`tests/test_assistant_js.py`): its path, method and
body, the trace's first and last entries with the scroll at the press and
the first move, the cap at sixty with the rest counted, a server that is not
there costing the drag nothing; the Chromium mouse test reads what the page
posted. The hold: a finger's and a pen's drag make the scrolling element
`overflow: hidden` with a stable gutter and the focused root too, put back
as found by the pointerup, a cancel and `dispose`, a mouse's drag holding
nothing (node); in Chromium a finger held on the launcher finds the root
hidden with its gutter kept and the account naming it, and the page scrolls
again once the finger lifts. Six mutations each fail a test: the scroll never
held, never let go, the gutter not kept, a mouse's drag held too, the scroll
never sampled, the focused root left scrollable. The journal
(`tests/test_assistant_journal.py`): one line that reads before its JSON,
the kind one word, what is not an account refused with its reason, the line
landing in `model_chain.log` between two others through `mc_logfile`'s own
handler, the route registered with the others and answering the file's
name. Each of the seven was reverted one at a time and a test failed: the
cancel, the finger's exemption, the `user-select`, the `touch-action`, the
touchmove wiring, the live-drag guard, the `dragstart`; four of the
account's: never written, nothing said, the scroll not read, the touch
moves not counted; and seven of the journal's: the page never posting, no
trace, no environment, the module logging nothing, the summary lost, the
route unregistered, a bad kind accepted. Run on the user's machine twice:
still moving the page the first time, and the second time with the log,
which is where the hold comes from. The hold itself has not run there.

### The account removed (2026-10-14)

At the user's word: "remove those flyout movement logs from all logs, i
should not see those drag events anymore". A drag now writes nothing
anywhere and keeps nothing: `watchDrag`, `accountForDrag`, the trace, the
page's state and environment, `forgeAssistant.shell.lastDrag`, the console
line, the status line's warning and the post to the journal are gone, and
with them the work the account did on every drag -- a trace entry per move
and the page's scroll read ten times a second, which flushes layout; at the
end the tabs' box, the selection, the document's size, two computed styles
and two media queries read, and up to sixty events serialised and posted.
What the account found stays built: the cancelled press, the handles' rules,
`blockPan` and the hold (`watchPan`, `holdPage`, `lockScroll`). The two
`describe` functions the script had declared at the same level -- the later,
the account's, was the one every caller got -- are one, with that body, so
the quiet-page watch names nodes as it did.

The journal (`mc_assistant_journal`) keeps one kind, the quiet-page watch's
`churn` (§3.36), and refuses an entry without a kind instead of taking it for
a drag. A `drag` entry -- which a page loaded before the update still sends
after every drag until it is reloaded -- is answered `{"ok": true, "file":
null}` and never written: a refusal would put a line on that page's console
after every drag instead.

Tests: in node a mouse's drag, a finger's with a touch move and the page
scrolled under it, and one the browser ended write to no console, post
nothing, tell the status line nothing and keep no account; in Chromium three
mouse drags (the last with the page scrolled under it) and a finger's drag on
a phone do the same; the hold's tests read what the hold took off the lock
itself (`scrollLock`) and, for a drag the page stays still under, watch the
root's inline style through the whole drag; the journal's failure path is
tested on the churn line. Nineteen mutations each fail a test: the console
line, the post, the warning and a kept account each put back (in node, and in
Chromium for a mouse and a finger), the hold never locking (node and
Chromium), a quiet drag locking for a moment at its end, `blockPan` cancelling
a move the browser took, the journal's refusal not noted, and on the server a
drag written again, the retired kind refused instead of answered, any
one-word kind kept, and the route refusing a stale page's drag.

## 3.35 Room for what a progress bar paints above itself

Asked for, with a screenshot of a run under the Model Chain theme: "I cannot
see the full animation for my progress bar. I have it set to ooze so the
animation should be above. How about this, when there is a progress bar,
there should be no button row for interrupt and skip. Instead, just the
progress bar the properly sized gallery and gallery buttons to fill below the
area needed to render the progress bar and 'ooze' above it. I want it to be
compact still, less space than what is currently dedicated to the row of
button and progress bar."

Reproduced in Chromium with the theme's page, the person's user.css, Forge
Neo's own rules for the bar and the progress-bar appearance set to Ooze:
Forge draws the bar absolutely, 14 px above the results panel, so it is not
a kept thing here (`inFlow`), and the panel one space under the theme's
header put the bar's top 6 px under the header and 40 px of the 54 px box its
bubbles rise in (`.mc-ooze-overlay`, 34 px above the bar) behind it. While it
ran the gallery started 76 px under the header, as at rest.

Two halves. The theme (docs/24 §0) takes the bar into the panel's flow while
it runs, 14 px tall, and does not draw the Generate box beside it -- the
layout is the theme's. The fill learns one thing, and it is not about the
theme: **a kept thing can paint above itself** (`reachOf`): a progress bar's
absolutely placed children that reach above its top -- the ooze's box --
measure how much room it needs clear. The first kept thing is lifted so that
room is under the ceiling (`lift` is the larger of what puts the frame one
space under the ceiling and what puts the first thing its reach under it;
`start` is where the first thing then is), and a kept thing after another
has that room under the one before it rather than one space. The room
overlaps the space and the frame's padding above the bar, which are empty,
so it costs only what it needs beyond them: on the theme's page the gallery
starts 56 px under the header while a run shows, against 76 at rest. A bar
with nothing above it reaches nothing, and is spaced as before.

Verified in Chromium: `tests/test_assistant_layout.py` (a bar carrying a 34 px
decoration, first: the decoration's top at the ceiling, the frame still one
space under it, one space between every kept thing after; and after
Generate: its reach under Generate, not one space) and
`tests/test_theme_layout.py` (the user's page during a run: the ooze whole
under the header and not covered, at rest and scrolled and under focus; the
row not drawn while Interrupt keeps Forge's display; one space between the
bar and the gallery, the buttons unmoved, the header-to-gallery distance at
most four fifths of the resting one; a press on the bar or the ooze toggling
the row, a control over the ooze and a press beside the bar not; the run's
end restoring the resting layout and the next run starting without the row;
a bar Forge hides keeping the row; a plain bar at Generate's resting place).
Fourteen mutations each fail a test: the reach at the ceiling, between
leaves, in `start`, or never measured; the bar left absolute; the row drawn;
a hidden bar hiding it; the press ignored by it; the bar or its fill at 20 px;
the press never wired, on the bar alone, counted on a control, or beside
the bar. Not run on the user's machine or in Firefox.

## 3.36 Steadier drags: what the page did on its own, and what a move cost

Asked for after #277: "dragging the fly out on the text to image tab has
worst performance than dragging it on a tab with simpler UI, like PNG info
... Is something on text to image constantly loading? Even image to image
tab is better handling. Maybe could it be this other extension?" -- with
`sd-gallery-quadrant` attached. Then, on the evaluation: "Let's do all 6, i
need smoothness. Make one giant PR with all 6 changes. I should see no UI
impact other than enhanced performance."

The attached extension is not the cause: one idempotent after-update
callback (dataset flags) and hover listeners on the gallery's preview. What
was, measured in Chromium against the user's own drag log:

### Two loops that never stopped

Forge's `script.js` observes its app element (`childList`, `subtree`) and
runs every `onAfterUiUpdate` callback 250 ms after any burst of changes. A
callback that changes a child list on every run therefore raises the next
run itself, forever, on every tab. Two of this extension's did:

- **The theme's header** (`installHeader`, `fit`): the title's text written
  on every apply -- the same words, a new text node -- and the ⋯ menu emptied
  and refilled on every fit. Twenty runs and sixty records in five idle
  seconds.
- **The spatial compact canvas** (`wireCompact` → `compactLoad` →
  `compactPaint`): the regions holder emptied and redrawn on every run
  whenever the saved layout held a region. Nineteen runs and fifty-seven
  records in five idle seconds; none with no region.

The user's log showed the result: under every drag, a stall of about 50 ms
every 300 -- the debounce plus one turn of every extension's callbacks, nine
of them this extension's, each costing what the visible tab costs to query.
Txt2Img is the biggest tab, the canvas's rewrites live in it, and PNG Info is
the smallest: hence the difference the user felt.

Fixed by writing only what changed: the title when it differs (`show`,
`setClass` for the attributes), the menu when the hidden tabs or their names
differ (`menuFor`, compared button by button); the canvas when the saved text
or the holder it was drawn into changed (`drawnFrom`, `drawnInto`). Zero runs
in five idle seconds, both; the menu and the region still drawn. And a watch
for the next one: the shell counts after-update runs (`watchChurn`,
`noteUpdate`) and, at twenty within ten seconds with nothing pressed for
three and no generation or reply running (`pageBusy`), writes one `churn`
line to the journal naming the nodes whose child lists changed most, once per
page -- where to look when another extension does it.

### The dock, 85 layouts a move

With the settings column docked, every move of the panel placed the column
again (`Dock.place`), and the guard's observer -- which refits every text box
when Gradio restyles one -- took the dock's own style writes for Gradio's and
ran `fitAll`: one forced layout per box. Measured: 85 layouts a move, 1,867 ms
of layout over 61 moves, about 30 ms a frame. Now a drag shifts the column by
the panel's own transform (`Dock.shift`, settled by the next `place`), the
dock tells the guard to forget its own writes (`Guard.forget` after every
placement), and `fitAll` sets every box to measure, measures them all, then
writes them all: 0 layouts and 0 fits over 30 moves of a docked drag; a
refit of 25 boxes 4 layouts instead of 25.

### The hold, lazy

§3.34's hold -- `overflow: hidden` and `scrollbar-gutter: stable` on the
scrolling element at a finger's press -- cost the whole visible tab a layout
at the press and another at the release: 8 ms on a 3,400-element Txt2Img,
65 on a 9,500-element one, under a millisecond on PNG Info, on every drag,
in browsers that never pan at all. (Measured apart, it is the gutter's
change that relays out the page, not the overflow's, 17 and 53 ms against
0.09; the gutter stays, because without it the page shifts by a scrollbar's
width under the handle on a browser with classic scrollbars.) The hold now
waits for the pan: `watchPan` listens for `scroll` on the document in the
capture phase while a finger's or a pen's drag is live -- the page's own
scroll arrives at the document, the focused root's at itself, a box's inside
the page is not the page -- and the first one holds the page (`holdPage`:
`lockScroll`, the page put back to where the press found it, once, and a pan
the browser keeps making against the hold not fought frame by frame). A
browser that honours the handle's word pays nothing. The account gains
`held: {t, by}` and a `hold` trace entry; `locked` names what was held, `[]`
when nothing was. (The account itself was removed later the same day, at the
user's word: §3.34, "The account removed".)

### A move by transform, and once-a-drag housekeeping

`moveDrag` wrote `left` and `top` on every move: a layout of the handle's
subtree and, in Gecko, a repaint of the whole panel, every frame. Now the
handle is moved by `transform: translate(dx, dy)` with `will-change:
transform` for the drag's duration -- a layer of its own, composited where it
lands -- and `endDrag` clears both before the placement writes the position
once. The launcher with a look is already `isolation: isolate` and the
panel's fixed children are top-layer dialogs, so the stacking context a
transform makes changes nothing visible. Per move, what cannot change under a
drag is read or written once: the safe-area insets (`drag.inset`), the
`forge-assistant-dragging` class, the viewport read once and passed along,
and the ghost drawn only when the anchor changes (`drag.previewed`).

### Containment, where it is safe

Measured on the heavy page with 300 messages in the transcript: `contain:
strict` on `#forge-assistant-root` (with `width: 100%; height: 100%`, the
same box as `inset: 0`, said as a size because Gecko takes a contained box
for a reflow root only when its size is written down) makes 20 transcript
appends 5.2 ms of layout against 7.5, with a page relayout and a drag
unchanged. Every child of the root is fixed to the viewport the root is, and
its dialogs are in the top layer, which containment does not reach. Not on
Forge's own columns, which the evaluation had offered: `contain: layout`
makes a column a stacking context and the containing block of its fixed
descendants, and a probe in Chromium showed it -- a 1,400 px dropdown list
from inside the settings column is hit by `elementFromPoint` without
containment and the gallery's `grid-wrap` is hit with it, the list painted
under the next column. A stylesheet test now forbids `contain` outside the
assistant's root.

### Verified

`tests/test_theme_layout.py`: applied three times to a settled 820 px page
with the ⋯ menu holding tabs, the theme makes no child-list change Forge's
observer would hear, and the menu still holds the hidden tabs; a changed
title and a renamed hidden tab still reach the bar and the menu (four
mutations: the title written every apply, the menu rebuilt every fit, the
menu never rebuilt, the title never rewritten). `tests/test_krea_spatial_js.py`:
wired twice more with nothing changed, the very same region elements are
still there and the note is unchanged; new saved text and a holder Gradio
rebuilt are drawn again (three mutations). `tests/test_assistant_layout.py`
in Chromium with CDP's layout counter: thirty moves of a docked drag measure
no text box and lay nothing out; twelve placements measure none and a new
width refits once; one `fitAll` of 25 boxes is at most six layouts (three
mutations, the unbatched `fitAll` among them, and the column placed on every
move). The hold: a finger's drag on a phone holds nothing while the page
stays still, holds it at the first `scrollBy` with the page put back to 0
and the account naming the hold, and lets go at the lift (node: nothing at
the press, held once with the start restored, a mouse watching nothing, the
focused root, a cancel, `dispose`; the listener on the document in the
capture phase, deaf to a box's scroll, removed with the drag; five
mutations). The transform: in Chromium the panel's inline transform is
`translate(-300px, 40px)` mid-drag with its position untouched, the column's
the same with `!important`, both empty and the position written once after
the release (node: the same, and `put` still the box's position; two
mutations). Housekeeping: the insets read as often for thirty moves as for
three, the class put on once, the ghost written for the first anchor and
four times for a new one (three mutations). The watch: twenty runs with
nothing pressed for five seconds report once with the nodes counted, a
twenty-first and twenty-five more report nothing, a press, a generation
(Forge's Interrupt shown) and a reply in flight (the store's operation not
terminal) each count from nothing again; in Chromium twenty-five bursts
against the settings column post one `churn` account naming
`div#txt2img_settings.column` and ten more post nothing (six mutations, and
the summary's on the server). Containment: the root's rule and no `contain`
elsewhere (two mutations). The full suite is green. Nothing of this has run
on the user's machine.

## 3.37 The tab bar in focus mode (2026-10-22)

Asked for once the Model Chain theme drew every tab as an icon on one bar
(handoff 49): "Let's make that entire row available in focus mode. So by
default, the tab bar should be available. I want options in the flyout menu
"..." for toggling the new thin tabbar to show / hide. Also when hidden, lets
make a tap of the flyout button or flyout menu drag area tap temporarily show
the tab bar."

### The bar kept, found by its button

Focus used to take every tab bar out of the layout with the rest of the
chrome. It now keeps one: the strip holding the button that shows the
focused workspace (`ownBar` in `forge_assistant_focus.js`, a child whose
`aria-controls` is the root's id), which is Forge's `#tabs > .tab-nav` and,
under the Model Chain theme, its header. Found by the button and not by a
name, for the reason every other bar is marked one at a time: `.tab-nav` and
`[role=tablist]` are on every nested tab group too. Kept only where its parent
is on the workspace's path; a bar a theme moved under some other parent sits
inside chrome the path rule takes out, and is hidden one at a time as it
always was. Never one that holds the workspace or the assistant.

The kept bar carries `forge-assistant-focus-bar` (the path rule leaves it in
the layout, with anything marked `forge-assistant-focus-keep`), and the body
says its mode, `data-forge-assistant-focus-bar`:

- `shown`, the default: fixed along the window's top, 8 px in from each side
  as the workspace's content is (the root's inset), a step above the
  workspace and below the assistant (`--forge-assistant-workspace-layer` + 2),
  on the page's colour; the focus root starts at the bar's bottom edge, which
  the module measures and writes as `--forge-assistant-focus-top` on the body,
  again whenever the bar changes size (a ResizeObserver: the theme's thin bar,
  Forge's own bar wrapping onto a second row).
- `hidden`: above the window's top edge (`translateY(-100% - 2px)`,
  `visibility: hidden`), and the root at the window's top. No slide either
  way: a transition also ran as focus came on and showed the bar leaving a
  workspace it was never meant to be over.
- `peek`: the bar over the workspace's top, the root where it was.

Nothing is moved in the DOM. A press on one of the bar's tabs is the host's
own; the shell's `subscribeNavigation` moves focus to the new workspace, and
`moveTo` keeps the bar for it. The mode is the person's, kept on the Focus
instance across focus going off and on (`setBar`, `placeBar`, `keptBar`);
the safety valve's plain overlay keeps no bar.

### The host reads Gradio's word in focus mode

A press on the kept bar first did nothing visible: focus stayed on the old
workspace. `getActiveWorkspace` asks which panel is *showing*, and in focus
mode that is focus's own doing -- the focused panel fixed and drawn whatever is
selected, every other one taken out by the path rule, Gradio's inline
`display: block` on the new one notwithstanding. So in focus mode
(`forge-assistant-focused` on the body) the host reads the panel Gradio's
inline display names, and the selected button after it; out of focus mode,
what is drawn decides as before. The same fix lets a switch made elsewhere
while focused -- Send to img2img -- move focus, which it never could. The
workspace picker's switch went round this by leaving focus first
(`switchWorkspace`), and still does.

### The ⋯ menu's switch, and a tap to peek

**Tab Bar in Focus**, a `menuitemcheckbox` in the Panel group after Free
Float, remembered in this browser (`forge-assistant-focus-bar:<base path>`,
on until turned off). Off, a tap on the launcher (`tapLauncher`, any press,
the keyboard's too) or on the header's space (`tapHeader`, at once, whether
or not more taps follow) brings the bar over the workspace for `PEEK_MS`,
4 s (`peekBar`, `armPeek`). Anything done on the bar or on what goes with it
-- a press, a move, a wheel, focus or a key inside, heard in the capture phase
and passively, and ignored unless the bar is peeking (`notePeek`) -- begins
the moment again. At its end the bar stays while a mouse rests on it or while
something in it holds a focus the browser draws (`peekHeld`): a finger's tap
leaves `:hover` on what it tapped and focus on the button it pressed, and
either would have held the bar for good, so hover counts for a mouse alone
and focus only under `:focus-visible` (the keyboard's, or a text box being
typed in). Focus going off ends the moment (`focusOff`, `endPeek`). A drag
of the launcher or the panel brings it too, since §3.46.

### What goes with the bar

The Model Chain theme's bar carries two drawers' buttons. Its quick settings
drawer lives beside Gradio's container and was taken out with the chrome; it
is marked `forge-assistant-focus-keep` now, and under focus hangs from the
bar's bottom edge (`--forge-assistant-focus-top`) a step above the workspace
and under the bar, and is not drawn while the bar is hidden. Its extra
networks drawer lives inside the workspace's own tab and was hidden in focus
mode; with the bar shown or peeking it opens as on the page, over the
workspace's right edge from the bar's bottom, in both of its layouts -- the
column layout's grid is not drawn in focus mode, where the workspace is its
own scroller and has nothing to stick under (docs/24 §0).

### Mini Paint NEO

The focus root no longer always starts at the window's top. Mini Paint's
WanGP tab sizes its frame from its column's top to the panel's bottom
(`fitFrame` in `browser/minipaint_wangp.js`), so a root that starts under the
bar is filled from there with nothing changed on that side; its tests carry a
copy of this stylesheet's focus rule (`FOCUS_STAND_IN_CSS`), which is to
follow the rule.

### Verified

`tests/test_assistant_focus_js.py` (node): the bar holding the workspace's
button kept and the root's top its bottom (44 px), a bar showing no workspace
and the workspace's own nested tabs as before; hidden and peeking the root at
0, a mode it does not know changing nothing; the choice outliving focus going
off and on; every mark gone on leaving; a bar under a parent off the path
hidden as before; the safety valve keeping none. `tests/test_assistant_js.py`
(node): the switch a remembered checkbox that tells focus, on until turned
off; a tap peeking only with the bar hidden in focus mode; the launcher (a
press and a key) and the header both peeking; the moment ending alone, held
by a resting mouse and not by a finger's hover, held by a typed focus and not
a pressed button's, renewed by what is done on the bar and not elsewhere,
ended by focus going off; in focus mode the host reading the panel Gradio
chose while the drawn one is the old workspace. `tests/test_theme_layout.py`
(Chromium, the theme's page with the assistant's scripts): the bar fixed at
the top, 44 px, 8 px in from each side, the workspace under it and a press on
a tab reaching the tab; a tab pressed on it moving focus to its workspace;
both drawers opening from its bottom edge over the workspace; hidden, the bar
and the quick drawer gone and the workspace the window, the menu's entry
unchecked, and both back when it is shown again; a launcher tap and a header
tap peeking it over the workspace, which stays where it was, and the moment
ending on its own; leaving focus putting the bar back in the page. The three
older tests that held focus mode's bar out of the layout now hold it with the
bar hidden and also measure it kept.

## 3.38 WanGP's prompt in the panel (2026-10-26)

Asked for from use: "I like how text to image has an option in the flyout
menu that pulls in the entire configuration into the flyout. I want a very
simple version of this, but for wan2gp. when I'm on that tab for generating
content, I would like the ability to directly modify the prompt from the
flyout menu just like image generation in text to image, but just the prompt."

### Why a box of the panel's own, and not WanGP's drawn under it

Txt2Img's column can be drawn under the panel because it is in this page
(§3.29). WanGP's form is not: it is a Gradio 5 page in Mini Paint NEO's frame,
on Forge's origin through its proxy, and nothing outside that document may
reach into it -- Mini Paint's bridge is the only thing that knows a WanGP
component (its `compatibility.py`), and its page script is the only thing in
that document that speaks to this one. So the panel draws a box of its own and
the bridge reads and writes WanGP's: bridge **1.14.0**'s `PROMPT_REQUEST`,
answered by `PROMPT_RESULT`, behind `minipaintWanGP.prompt()` and
`setPrompt(text)` (Mini Paint's `docs/wangp/CONTRACTS.md`, *Bridge 1.14.0: the
prompt*). The box is the one WanGP generates from -- `wizard_prompt` while
WanGP's prompt wizard is on, `prompt` otherwise, the choice a queue request
already makes -- and a write is that box's own value from then on, through the
bridge's one Gradio event as an output: not a queue request's overlay, so
nothing puts it back, and WanGP's page shows it as if it had been typed there.
The text crosses as typed (`normalize_prompt_edit`): line endings made `\n` and
control characters removed, never stripped, and an empty box is a prompt
cleared.

### The panel's side

The same 🎛 button, on the WanGP tab, named *WanGP prompt* there
(`applySettingsButton`); a third state of its own, `wangpPrompt`, beside
`settingsDocked` and saved with it, so a view chosen on one tab never changes
the other's (`promptShown`, `toggleDock`; `toggleChat` leaves it as it leaves
the dock). Its view, `.forge-assistant-wangp`, takes the conversation's place
as the dock's placeholder does (`applyChat`, `applyMenu`), on the panel's own
surface -- not `forge-assistant-docked`, whose see-through panel lets presses
through it to a column that is not there here. It needs a Mini Paint NEO
bundle with both functions; without them the button is not offered on the
WanGP tab.

*Reading.* `applyPrompt` reads WanGP's page when the view is shown (the press,
the WanGP tab coming back with the view on, the panel opening) and on the
panel's five-second beat only when `state().prompt.touched` -- the WanGP page's
own change notices, a person's hand on its form -- has moved since the last
answer: one question per touch, never one per beat, a refusal included. A read
never lands over text the box holds that WanGP has not got (`_promptDirty`,
`_promptSending`), and only one is in flight.

*Writing.* `PROMPT_QUIET` (400 ms) after the last keystroke, at once on blur and
when the view is put away (`writePrompt`); one write at a time, and text typed
while one is on its way gets a write of its own after it. A write Mini Paint
counts as a change to WanGP's form (its record is saved a second later, as
after a touch) and not as a touch, so the box is never sent back to read its
own writes. A write WanGP did not take keeps the text as unsent and says why
under the box.

*Generating.* ▶'s pump settles the box first (`settlePrompt`): a press waits
for what was typed to be in WanGP's page before `generate()` asks WanGP's own
Generate to read it, and a press whose prompt was not taken generates nothing
and says "WanGP's prompt was not updated: ...". Ctrl+Enter (or Cmd+Enter) in the
box is ▶, with the key's default prevented (Enter alone is a new line, which
WanGP reads as one more prompt).

*Shut.* While WanGP's page has not answered the box is greyed with "WanGP is not
loaded yet."; under a bridge whose READY did not say `prompt_edit` -- one older
than 1.14.0 -- with "Reinstall WanGP's bridge from the WanGP tab to edit its
prompt here.", and nothing is asked in either case. A page whose prompt is over
the ceiling (12000 characters) is not read out, and the box says to edit it on
the WanGP page.

### No live connection

Asked for once it was built: "Make sure there is no live stream. No live
connections!" The panel opens nothing for the box: every read and write is a
call into Mini Paint's bundle, which posts a message into WanGP's frame and
waits, with a deadline, for the one that answers it -- no `fetch`, no stream,
no socket, no repeating timer of the box's own, and after the box has been
typed in, written, generated from and put away nothing of its own is left
set. The panel's five-second beat was already there (the free-memory reading's
plain request with its deadline, §3.26) and asks WanGP nothing for the box
unless WanGP's form was touched by hand. Inside WanGP's page each read or
write is one Gradio event, as every bridge operation is: on Gradio 5.29,
WanGP's pin, a `queue/join` POST and a `queue/data` result stream that Gradio
closes as soon as the event answers -- about 60 ms on a 5.29 page carrying the
real bridge's controls, event, page script and plugin handler, driven in
Chromium; at rest the only thing open there was Gradio's own heartbeat, open
from the page's load before any prompt operation and WanGP's, not the box's
(Mini Paint's `docs/wangp/CONTRACTS.md`, *Bridge 1.14.0: the prompt*).

### Verified

`tests/test_assistant_prompt_js.py` (node, Mini Paint's bundle a stand-in of
its contract): the button on the WanGP tab only with both functions, named for
it; a press showing the box in the conversation's place and reading the page,
nothing drawn under the panel; a second press, Chat and the tab bar going back;
Txt2Img's column and WanGP's view each the other's no business; one write once
the keys go quiet, at once on blur, after the one on its way, and before the
view goes; ▶ writing first and then generating, generating nothing when the
write was refused, asking nothing first when nothing was typed; a read never
over unsent text, again only after a touch by hand, afresh when shown again;
the box shut with its reason and asking nothing, the wizard's box named, a
refused read shutting it until WanGP is touched; Ctrl+Enter; the state kept
with the layout; and the whole flow opening no stream, socket or request,
setting no interval and leaving no timer of the box's own. Seventeen mutations
each fail a test.
`tests/test_assistant_wangp_prompt.py` (Chromium, the layout test's page with
a WanGP tab): the box drawn inside the panel, on its surface, taking a press;
typing reaching `setPrompt` once the keys go quiet through the real wiring; ▶
straight after typing and Ctrl+Enter generating with what was typed; on
Txt2Img the button its own, and back on WanGP the page read again; from the
page's load through typing, ▶ and Ctrl+Enter no WebSocket and no event stream,
and every request finished. Mini Paint
NEO's halves are its `tests/test_wangp_queue.py` (the plugin's operation),
`tests/test_wangp_protocol.py` (`prompt_checks`, the bundle on its virtual
clock, nothing opened or left pending, and the WanGP document's script) and
twelve mutations. The bridge's prompt read and write ran once on a real Gradio
5.29 page, as above; not against a real WanGP or on the user's machine.

## 3.39 Three modes, and Super Focused on the WanGP tab (2026-10-26)

Asked for once §3.38 had run on the user's machine ("Okay, that worked
well"), in three parts:

- "The text in the new box does not seem to properly handle new line. It
  strange, in wan2gp prompt box i see new lines, but in the flyout there is no
  space at all between lines."
- "i really like that i can resize the flyout menu for the text field ... i was
  worried at first when i opened the flyout prompt view for wan2gp and it was
  really small."
- "a new mode that is just the video gallery from wangp (currently playing and
  strip) with playback controls, and the prompt box with a generation button,
  and progress bar. The view should try to layout in two columns that can be
  resized, with prompting on left, and gallery on right, sort of like text to
  image tab in forge neo! ... if I double tap the flyout button action
  activation / drag handle, it cycles through Normal Mode / Focused Mode /
  Super Focused Wan2gp Mode. Super Focused mode would just be those elements,
  and when i return out of it, everything and all settings return (except
  updated prompt and gallery)."

### The box's lines and its size

Not reproduced as reported: every hop keeps a newline -- WanGP's page through
Gradio 5.29 and the bridge (driven on a real 5.29 page), Mini Paint's bundle,
the panel's `readPrompt` -- and Chromium and a Gecko build (Zen) draw the box's
lines as lines under the panel's stylesheet. Two things could still close them
up on the user's page, and both are now the box's own: the box said `font:
inherit`, which brings the page's *line height* with the face -- a line height
a theme or a page sets as a length for smaller text is less than a line at the
panel's text size, and the lines touch -- and a theme may say how text areas
wrap. `.forge-assistant-wangp-prompt` says `line-height: 1.5` and `white-space:
pre-wrap`, both `!important` (a class outranks a theme's `textarea` among
important declarations), and `overflow-wrap: anywhere`. The box starts
`PROMPT_ROWS` (8) lines tall rather than six, at least 6em; a drag of its corner
is the browser writing an inline height, which a `ResizeObserver` hears and
keeps in `localStorage` under `forge-assistant-wangp-prompt-height:<base>`
(`watchPromptSize`) -- read and remembered, never written back, so the
observer cannot answer its own change -- and the box is built at that height
(`_restorePromptHeight`), the stylesheet's 60vh still the ceiling.

### Two presses, three modes

`tapLauncher` and `tapHeader` count presses as before (`TAP_WINDOW`), and the
second press now acts inside itself, as the third did, because entering focus
asks for the browser's full screen and Firefox grants that only while a press
is being handled: `cycleMode`. A third press is the first of a new count.
Normal goes to Focused (`toggleFocus`); Focused goes to Super Focused where
there is one (`enterStudio`) and to Normal where there is not; Super Focused
goes to Normal. Going back to the previous workspace, which two presses did
since round one, is the workspace menu's first entry, **Back to** the
workspace before this one (`backItems`, `backToPrevious` unchanged), shown
only while there is one.

### Super Focused

It is the WanGP tab's (`studioRefusal`): elsewhere, or without Mini Paint NEO's
bundle, there is no third mode and nothing is said; a bridge whose READY did
not say `layout_studio` (older than 1.15.0) skips it with "Reinstall WanGP's
bridge from the WanGP tab for Super Focused mode." on the status line; a WanGP
not loaded yet has it, because Mini Paint tells the page its layout at its
ready.

Here it is one class beside focus's on the focused panel,
`forge-assistant-wangp-studio` (`focus.setStudio`; `exit` takes it off with
focus's own, so no path out of focus can leave it), which Mini Paint's WanGP tab
already watches the panel's classes for: it sends `LAYOUT_STATE {compact:
true, studio: true}` and the bridge lays WanGP's page out (Mini Paint's
`docs/wangp/CONTRACTS.md`, *Bridge 1.15.0: the studio*). The class's name is a
contract with that extension, like the focus root's. `enterStudio` notes what
it puts away -- whether the panel was open -- in `state.studioReturn`, closes
the panel and hides the tab bar whatever *Tab Bar in Focus* says
(`applyFocusBar`; `peekBar` peeks nothing). `endStudio` puts the panel back as
it was, an open panel opened again and one opened over the studio closed, and is
called from `focusOff` -- the path two presses, Escape, the menu's exit and the
browser leaving full screen all take -- and from `navigated` when the
selection moves off the WanGP tab (focus then follows the switch as it always
has). `studioReturn` is saved with the layout, so a reload in Super Focused
(which, like every reload, starts outside focus) opens the panel as it was
before. The panel says `data-studio="on"` while it lasts.

### The studio, in WanGP's page

Mini Paint's half, recorded there and summarised here. Nothing in WanGP's page
moves: the generator form's row becomes a grid, its two columns `display:
contents`, and the blocks kept are placed in it where they already are in the
document -- the prompt boxes (found by the ids wgp.py gives them, with the
wizard's variables between), WanGP's action bar (Generate, or One More, Pause,
Abort and Add to Queue while it generates) and its progress in the left column,
the gallery tabs (the video playing with its controls, the strip under it) in
the right, spanning every row, the gallery itself as tall as the window leaves
it. Every other block of the form, what WanGP has around it and Gradio's footer
are taken out the compact layout's way; every declaration written is inline,
`!important`, and recorded with what was there, so whole puts each back as it
was. A divider of the bridge's own between the columns is dragged, moved with
the arrow keys or reset with a double click, and its share is kept in this
browser; under 720 px the columns stack. The page answers what it settled on
(`LAYOUT_RESULT`), and Mini Paint's journal says when that is less than was
asked: no prompt box (compact), no Generate bar or progress (a studio without
it).

### Verified

`tests/test_assistant_modes_js.py` (node): the cycle on the WanGP tab
(Normal, Focused, Super Focused, Normal, with the panel, the bar, the mark and
what was put away at each step) and elsewhere (no third mode); no bundle, an
older bridge (skipped, with the sentence) and a WanGP not loaded yet; a closed
panel staying closed; Escape's and the full screen's path, a switch of tab and
a blank read of the selection; a reload; the bar and its peek; the workspace
menu leading with Back to. `tests/test_assistant_taps_js.py` translated: two
presses act inside the second, a third starts again, the header's space the
same, going back never two presses. `tests/test_assistant_focus_js.py`: the
class on the focused panel and off it, with focus and on a move.
`tests/test_assistant_wangp_prompt.py` (Chromium): a two-line prompt read into
the box as two lines, the lines a line and a half apart and three more lines a
box three lines taller, under a theme that says `white-space: normal` and an
11 px line height about every text area, both `!important`; the box eight lines
tall to start, its corner dragged, the height kept and the box that height
after a reload; two presses on the header's space, two more, two on the
launcher -- Focused, Super Focused with the panel put away and the bar hidden,
Normal with the panel back; Super Focused left by Escape with everything back.
`tests/test_assistant_layout.py`'s finger test is two taps now. Sixteen
mutations each fail a test. Mini Paint NEO's halves: its
`tests/test_wangp_protocol.py` (`layout_checks`: the vocabulary, the studio
scenario on the virtual clock, an older bridge told compact) and
`tests/browser_intercept.py` (`check_super_focus_makes_the_wangp_page_a_studio`,
the bridge's own script on WanGP's page as Gradio 5.29 draws it, captured,
under a hostile theme; six mutations). The studio was also watched on a live
Gradio 5.29 page shaped like WanGP's form, with WanGP's own stylesheet and
page script and the real bridge, through a generation: the progress moving,
the live controls in Generate's place, a new picture in the gallery, the
layout holding, and every declaration put back. Not against a real WanGP or on
the user's machine.

### The studio laid out again (2026-10-27, Mini Paint's bridge 1.15.1)

The user's first run, in two parts:

- "It goes into single column mobile optimized way too soon. Let's reduce the
  limit, so that single column narrow view only happens on narrower displays
  like a phone. When I 120% zoom on my fold 7, it went into single column"
- "Left column should just be the prompt field, the entire column is just the
  prompt field which can resize by moving the divider. If too much text, than
  I can scroll the input field. The right side should have the generate and
  progress bar, above the gallery."

All of it is Mini Paint's (its `docs/wangp/CONTRACTS.md`, *Bridge 1.15.0: the
studio*, which says what 1.15.1 changed); nothing here changed -- the class,
the cycle and the refusal read `layout_studio`, which both bridges offer. The
left column is WanGP's settings column with the prompt the one thing left in
it, across every row of the grid, and the prompt's box is as tall as the column:
its height comes from a stylesheet rule of the bridge's (`!important`, keyed on
a custom property the studio writes on the row) because Gradio writes the box's
height inline as it is typed in, and it scrolls inside itself. The right column
is Generate, then the progress, then the gallery tabs, the gallery taking what
the window leaves; when WanGP's live controls take Generate's place the studio
is fitted again (a `ResizeObserver` on what stands above the gallery and on the
prompt's blocks, never on what it sizes), so the gallery gives the room back and
the two columns still end together. The columns stack only below **520 px** --
a phone -- where the prompt's box is 30% of the window; above it each column
keeps a minimum (200 px for the prompt's, 280 for the gallery's), so a Fold 7's
inner screen at 120% (about 690 px) is two columns. A stacked page starts at
its top, and leaving the studio puts the page's scroll back where it was.

Verified there: `check_super_focus_makes_the_wangp_page_a_studio` rewritten for
the new shape (the prompt alone on the left, its box ending where the gallery
does; Generate over the progress over the gallery; typed past its height under
Gradio's inline height and a theme's cap, the box keeps its size and scrolls;
the action bar grown by 60 px, the gallery shrinks and both columns still fit;
two columns at 700 px, stacked at 480; the scroll at the top and put back),
thirteen mutations each caught; and on a live Gradio 5.29 page with WanGP's own
stylesheet and page script, through a generation and at 1400, 680 and 400 px.
Not against a real WanGP or on the user's machine; the bridge needs
reinstalling from the WanGP tab.

## 3.40 Two presses back, three focus, four Super Focused (2026-10-27)

Once §3.39 had run on the user's machine:

- "I noticed that doube tap flyout to switch to last tab no longer works. I
  think you updated handling because of super focus mode. Let's fix this."
- "Let's make double tap switch to last tab, triple tap open focus and 4 taps
  open super super focus."
- "Please treat super focus the same as focus mode, meaning if I switch to
  another tab, then I'm still in focus mode, but when I return back to wan2gp
  It should be in either focus or super focus, whichever I left it off on."

The diagnosis was right: §3.39 made two presses the mode cycle (`cycleMode`)
and moved going back to the workspace menu, which with the panel put away in
super focus is out of reach.

### The counts

The launcher and the header's space count the same way (`tapLauncher`,
`tapHeader`), and the decisions are shared (`pressThree`, `pressFour`,
`afterPresses`):

| Presses | Acts | Does |
|---|---|---|
| 1 | when the window passes | the launcher opens the panel; the header's space does nothing |
| 2 | when the window passes | back to the workspace before this one (`backToPrevious`) |
| 3, focus off | inside the third press | focus on -- the browser's full screen is granted only to a press being handled |
| 3, focus on | when the window passes | focus off: a fourth press may still come, and four mean super focus, not out |
| 4 | inside the fourth press | focus on if it is not; on the WanGP tab Super Focused on, or off again to Focused; elsewhere Focused |

A press after four starts a new count. On the WanGP tab with a bridge that
cannot lay the page out, four presses leave focus on and say to reinstall, as
the cycle said it. Two presses' going back is what it was before §3.39 (on a
timer, the panel never opening), and the workspace menu's **Back to** stays.

### Super focus is focus, on the WanGP tab

`studioKept` is the wish, set by `enterStudio` and kept for as long as focus
lasts. A switch away -- going back (`switchWorkspace`), or any switch the host
reports (`navigated`: a tab on the kept bar, a picture another extension sends
to another tab) -- puts the studio away with `endStudio(true)`: the class off,
the panel and the bar back as ordinary focus has them, the wish kept. A switch
that lands focus on the WanGP tab brings the studio again (`resumeStudio`, from
`refocus` and from `navigated`), the panel put away again and what it was noted
again. The wish goes with focus (`focusOff`: three presses, Escape, the browser
leaving full screen, a switch focus cannot follow) and with four presses back
to Focused. Mini Paint needs nothing: its tab lays WanGP's page out from the
panel's classes, whole when focus moves away and a studio when it comes back
(and since bridge 1.15.1 puts the page's scroll back each time).

### Verified

`tests/test_assistant_taps_js.py` (node, a fake clock): two presses go back
when the window passes and not before; three turn focus on inside the third,
or off once the window passes; four act inside the fourth and, in focus, never
leave it; a fifth starts again; the header's space the same, one press
nothing. `tests/test_assistant_modes_js.py`: three and four on the WanGP tab,
four again back to Focused (the wish gone), three in super focus out of both,
four elsewhere Focused, no Mini Paint, an older bridge, a WanGP not loaded
yet, a closed panel; a switch away is focus with the panel and the bar back
and a switch back super focus again, through `navigated` and through going
back; plain focus comes back plain; focus ended on another tab forgets it.
`tests/test_assistant_wangp_prompt.py` (Chromium, real presses): three on the
header's space, four, four on the launcher, three; four on the launcher from
Normal; two on the launcher in super focus to the tab before, in focus, and
two on the header's space back to super focus; a tab pressed by the page and
the kept bar's WanGP button; Escape. `tests/test_assistant_layout.py`'s finger
test is three taps again. Thirteen mutations each fail a node test, and the
three on the switch's paths a Chromium test too. Not run on the user's
machine.

## 3.41 Img2Img's settings in the panel, a window you resize by its corner, and ▶ on the generation tabs (2026-10-27)

Three asks in one message:

- "I really like the feature in the fly out menu where I'm able to take all of
  the settings from text to image and force them into the fly out menu. Let's
  add the same ability for the image to image tab. I want all of the settings
  from the left panel to appear in the flyout menu, and only the generation
  content from the right panel should remain. It should be just like the text
  to image tab in regards to being able to see a full gallery view in the tab,
  And all of the settings to be configured appear within the flyout menu."
- "Additionally, for the flyout menu, I would like for the conversation view to
  be within a resizable flyout window so that I can resize it."
- "And last, for the flyout menu in both the text to image and image to image,
  I would like to have a play button, just like I do for WAN2GP, and its
  function should be to initiate generate".

### Img2Img's column, docked

Forge Neo builds Img2Img as it builds Txt2Img (`modules/ui.py`: a
`ResizeHandleRow` holding `#img2img_settings` and the results column from
`create_output_panel`), so §3.29 applies whole; only the dock knew one tab.
`Dock` keeps a column per workspace now (`DOCK_COLUMNS`: `tab_txt2img`,
`tab_img2img`; `docks`, a map), each with what it replaced on it, and
`place(rect, workspace)`, `release(workspace)` and `isDocked(workspace)` take
the workspace (Txt2Img's when a call names none, so the old calls mean what
they meant). One guard, on the column drawn in the panel: only one is ever
shown, and a column docked out of sight keeps its text boxes as they were
until it is drawn again. `shift` carries the column in the panel.

The panel's state is per workspace too, as WanGP's prompt is its own
(`DOCK_STATE`: `settingsDocked` for Txt2Img under the name it was always saved
by, `img2imgDocked` for Img2Img). `syncDock` goes through both: a column whose
state is off goes back to its page, the one on screen is drawn in the slot,
any other docked one is held out of its page out of sight -- the screen's last,
so the guard ends on it. Chat and 🎛 change the tab on screen's alone. Forge's
mobile script asks the same `offsetLeft` of `#img2img_results` as of
Txt2Img's, so `besideOf` and its pixel apply unchanged; the fill already
measured both tabs; the theme's one scroller already stood down for any
column carrying the dock's class.

**The canvas.** Img2Img's column holds what Txt2Img's does not: Forge's image
canvas (ForgeCanvas, `.forge-container`, for the img2img picture, Sketch,
Inpaint and the rest), which says `touch-action: none`, pans or paints on every
drag and zooms on the wheel -- 512 px of it in a view about as tall, so a
finger on it could never scroll the column past it. It is guarded as the
compact spatial canvas is (§3.31): in `GUARDED` and the stylesheet's list, a
field of its own (`FRAMES`, whatever block it is in, its brush's sliders part
of it), engaged by a tap. And because a picture is dropped onto it -- a file,
or one dragged from the gallery now beside it -- a drag carried over a guarded
field engages that field (`Guard.dragOver`, trusted events only), so the drop
is hit-tested onto the canvas and not onto the block a guarded field leaves
its presses to.

**A defect found on the way, older than this.** The rule giving an engaged
canvas its presses back was written as `.forge-assistant-guarded
.forge-assistant-engaged.mc-krea-spatial-compact-frame` (0,3,0) beside the
rule taking them away, `.forge-assistant-guarded :is(...):not(.gradio-dropdown
*)` (0,3,1): the lighter rule lost, so an engaged compact spatial canvas never
took a press either. Both canvases now have a rule of their own at (0,4,0); a
Chromium check reads both states' computed `pointer-events`.

### A window you resize by its corner

The only way to resize the panel was an invisible 8 px strip on its left
edge, width alone, 320 to 640. The panel has a corner now
(`.forge-assistant-corner`), drawn -- a triangle of hatching, its hit area the
triangle alone through `clip-path` -- at the corner away from where it is held
(`cornerSides`, written as `data-corner` at every placement): opposite the
anchor's corner when anchored, away from the window's nearest corner when
floating, and on the phone's sheet a grabber in the middle of the free edge,
height only. A drag of it (`startResize` with the corner; the edge is the same
machine) makes the panel as wide as the hand says (`panelWidth`, now up to 960,
the window's width still capping it; a panel centred across the window grows
both ways, twice the hand's travel) and the conversation as tall
(`conversationHeight`): one variable on the panel,
`--forge-assistant-conversation`, which the stylesheet makes the transcript's
height and max-height under `forge-assistant-sized`. The docked settings
column's view is the conversation's height by §3.29's rule (`sizeDock` reads
the transcript's max-height), so the corner sizes it too, from the corner in
that view. The conversation never goes past what the panel's cap leaves for
it, nor below 120 px -- or below the height it was drawn at when the corner
was taken, if that was less (an empty conversation is 0 tall, and starting
from the floor or the stylesheet's cap jumped it at the first pixel); dragged
back to nothing it is the stylesheet's again. The cap is read in pixels only:
the sheet caps itself as a percentage of the window, which a computed style
reports as written, and the first build read "100%" as a hundred pixels and
let the sheet's grabber move nothing. A double press puts the panel's own
size back (`resetSize`); the arrows move the focused corner. Not in the
tab bar, WanGP's prompt view (its box has a corner of its own) or under a menu:
there is no conversation to size. The size is the panel's layout, kept with
it for the browser tab, as the width always was.

Floating, the panel's place is a fraction of the room around it (§3.12), so
growing it moved every edge, the corner too. While the corner or the edge
drags it, `placePanel` holds the edges they are not moving (`heldEdges`,
`sizeHold`) and works the fraction back out from them; the fraction is saved
when the hand lets go. The corner the drag began on is kept through it, though
the panel's middle crosses the window's.

### ▶ on Txt2Img and Img2Img

The header's ▶ (§3.27) is the tab's own Generate on Txt2Img and Img2Img
(`FORGE_GENERATE`, `pressGenerate`, `generateHere`): `#txt2img_generate` or
`#img2img_generate` pressed, as Send to Generate presses it (§3.21), so what
that button does with the page -- the pipeline, the literal boxes, Img2Img's
mode -- happens because it is the button. Shown while the tab has its
Generate; never locked. Not twice: while the tab is generating, Forge covers
its Generate with Interrupt and Skip, so a person cannot press it, and the ▶
does not either -- the press is refused with *Txt2img is already generating.*
on the status line and, red for three seconds, on the button. Forge has no
queue of a page's own to put a second press in, which is why it does not line
presses up as WanGP's does.

One button for three tabs: what it last said is the workspace's it said it on
(`data-for`), cleared when another workspace shows; WanGP's presses still in
line go on, painted only on WanGP's button (`paintGenerateLine`) and again on
return.

### Verified

`tests/test_assistant_dock_js.py` (node): the button on Img2Img docking its
own column, each tab's state across switches in both directions (the screen's
column placed last, the other held out of sight), Chat giving back one, the
state saved; the dock itself with two columns (the guard following the one
drawn, a drag carrying it, one released exactly as found while the other
stays); the canvas a guarded field with its slider; a drag engaging the field
under it, a script's not. `tests/test_assistant_generate_js.py`: the ▶ on
each tab with its Generate, a press pressing it, a press during a run refused,
WanGP's line kept off another tab's button. `tests/test_assistant_resize_js.py`:
which corner for each anchor, floating and on the sheet; a drag sizing both,
its bounds, a centred panel, the docked view's height, the edge still only
widening, an empty conversation, the arrows and the reset, what is saved.
`tests/test_assistant_layout.py` (Chromium, an Img2Img tab on the Forge-shaped
page with a canvas that zooms on the wheel and counts presses): Img2Img's
column over the slot, scrolling, the gallery the row's width, held out of
sight on Txt2Img and back, Chat; a wheel over the docked canvas scrolling the
column until a tap engages it; a picture dragged from the gallery reaching
the canvas; the ▶ pressing each tab's Generate; the corner drawn top-left,
dragged by (100, 120) with the right and bottom edges held, the docked view
following, hidden in the tab bar, reset by a double press; a floating panel
keeping its other edges; on a phone's sheet, a finger drawing the grabber in
the middle of its top edge up 80 px. Thirty-four mutations each fail a test.
Not run on the user's machine, against Forge's own canvas, or in a Gecko
build.

## 3.42 A tap outside puts the panel away (2026-10-28)

Asked for once §3.41 had merged:

> "Let's make it so if the flyout menu is open, tapping anywhere outside of it
> will close it. It should close back to the button state. Additionally, the
> location of the flyout button when I tap out to close should be the closest
> docked location. The docked locations are those when free float is off,
> there are six of them. Dragging or scrolling outside of the open fly out
> should have no impact, only tapping / clicking outside of it should make it
> minimize to the flyout button. Make the new tap to close function enabled /
> disabled via a new "..." menu entry"

### What a tap is

The rule the theme's quick settings drawer already goes by (docs/24 §0,
2026-10-19; `tapOutsideQuick`): a tap is the `click` that ends a press made
outside the panel with the pointer no further than `TAP_OUTSIDE_PX` (16) from
where it went down (`notePressOutside`, `tapOutside`). What that leaves out:

- a scroll or a pinch: a touch the browser takes for a scroll is cancelled
  and makes no click, and a second finger is not primary;
- a drag across the page: it strays past the distance;
- a press on a scroll bar (`onScrollbar`, the page's or a box's): Chromium
  sends no click after one, and the press is not even recorded, for a
  browser that might;
- a press that began inside the panel (a selection, the panel's own drag or
  resize let go outside it), and a press just outside let go over it (the
  click goes to the two targets' common ancestor; where the hand came off
  says otherwise);
- the page's own clicks (synthetic: Forge's, the theme's, Mini Paint's), a
  button other than the first, and a key between the press and the click (a
  key that clicks a button has no press of its own).

The listeners are on the window in the capture phase (`listenForTaps`), so
nothing on the way can stop them -- the Gallery Quadrant stops a tap on the
gallery in the bubble phase -- and nothing is taken from the tap: what it
landed on still gets it. So the panel is put away with `showOpen(false)`, not
`close()`: the ✕ gives the launcher the keyboard's focus, and here the tap has
just given it to what it landed on. (**Replaced by §3.45**: the tap is taken
from the page and lands nowhere.)

**Not outside**: the assistant's root (the panel, the launcher), a docked
settings column (drawn over the panel's body, living in Forge's page), and
any dialog -- this extension's editors are native modal dialogs on the body,
and a dialog anybody opened is where the hand is meant to be (`NOT_OUTSIDE`,
the same list `noticeCover` spares).

**Frames.** A tap in a frame never reaches this document, and Mini Paint
NEO's WanGP tab is a frame on Forge's own origin. Every frame whose document
this one may reach is listened to as well (`watchFrames`, `watchFrame`: when
the panel opens, on a change of workspace and on the panel's beat, once per
document, and again when a frame loads another); a frame from another origin
cannot be reached and is left alone. A click ends only a press of its own
document.

### Where the launcher comes back

**Replaced by §3.44**, at the user's word: the launcher comes back at its own
place, and nothing moves. What follows is what this section built.

At the nearest of the six resting places (`closeFromOutside`). Anchored, the
panel is at one already and the launcher takes its place, as after the ✕.
Floating, `nearestAnchor` is asked with the panel's own box and middle, and
the answer is written as the floating place exactly on it
(`anchorFraction`: 0 or 1 against the edges it names, 0.5 across the middle
-- the launcher's travel is its whole box, so these are the anchor's own
pixels) and as the anchor too, both saved: Free Float stays on, the panel
opens at that place next time, and turning Free Float off leaves it there.
The ✕ is unchanged: it leaves the launcher where the panel was.

### The switch

**Tap Outside to Close**, in the ⋯ menu's Panel group after Tab Bar in Focus
(`tapToCloseItem`, `setTapToClose`): a `menuitemcheckbox`, on until turned
off, remembered in `localStorage` under a key of its own like Free Float's
(`_restoreTapToClose`, `_saveTapToClose`). It is read at the press; turning it
off spends any press waiting.

### Verified

`tests/test_assistant_tap_close_js.py` (node): a tap closes without moving
the focus; each of the cases above is no tap; the panel, a docked column and a
dialog are not outside; floating, the six places from four panel positions,
anchored nothing moved; the switch's place in the menu, its storage and its
default; the wired listeners spending a press on a cancel and on a key.
`tests/test_assistant_layout.py` (Chromium): a tap on a page button closes
the panel and the button is still pressed, a tap on a text box leaves it the
keyboard; a drag, the wheel, a script's click and a press from the panel let
go outside leave it open; a press on a scroll bar that takes room (in a
Chromium started without Playwright's `--hide-scrollbars`) records no press
and scrolls its box; on a phone a finger's scroll leaves it open and a
finger's tap closes it; the docked column and a modal dialog are not
outside; floating, the launcher lands exactly on the top-left and the
bottom-centre resting places and the panel opens there; the menu's switch off
and on; a tap in a same-origin frame closes it and a drag in it does not, with
the panel's beat turned off, from the panel opening and from the frame loading
again while it is open. Three menu-list tests translated to the new entry.
Twenty mutations each fail a test. Not run on the user's machine, in a Gecko
build, or against Mini Paint NEO's real WanGP frame.

## 3.43 No ✕, one tap on the header, and a drag that keeps the panel on the window (2026-10-28)

Asked for on the same branch as §3.42, to go in the same pull request:

> "when tap out to close is enabled, the x button in the flyout menu should be
> removed. The close action at that point is the tap out gesture. Also, when
> fly out is open, let's make single tap of the drag handle are perform the
> minimize as well. So another way back to the flyout button state."

> "I want intelligent drag when the flyout menu is open. Such case, if I drag
> to the edge of the screen and the flyout menu is now off screen, then it
> should resize so the edge that was going to be off screen is now in screen.
> essentially dynamically size the window so that it always appears in the
> screen view. even when I drag it and try to make it off screen, then that
> just means the draggable area appears in screen as well as the rest of the
> window. This should fix annoying bounce reset back to position I get when I
> drag it to the edge. Also, make the drag handle hit are larger. Its too
> small now."

### The ✕, one tap, and Escape

`applyMinimize` hides the ✕ while `tapToClose` is on and shows it again when
the switch is turned off. One press on the header's space, once the window for
a second has passed, puts the panel away (`afterPresses` is handed `putAway(true)`
for the header, as it is handed `open` for the launcher): back to the launcher
at the nearest of the six resting places (at its own place since §3.44),
exactly as a tap outside does
(`putAway` is that tap's code, `closeFromOutside` now calling it), with the
launcher given the keyboard's focus as the ✕ gave it -- the tap was the
panel's own. Two, three and four presses are what they were.

Without a ✕ a keyboard had no way to put the panel away, so Escape inside the
open panel does when the switch is on (`escapeOrder`: `close-panel`, after a
menu's claim and focus mode's, and only inside the assistant; elsewhere it is
Forge's, which interrupts a generation). With the switch off and the ✕ back,
Escape is what it was.

### A larger handle

The whole header row is the handle now: a press on one of its buttons may
start a drag (`startDrag`, `onButton`). Nothing is taken from such a press
until it has moved past DRAG_THRESHOLD -- no pointer capture (a captured
pointer's click goes to the panel, not the button), no default cancelled, not
counted as a press on the header's space -- so a tap stays the button's; once
it moves, the pointer is captured, the panel moves, and the drag's click is
swallowed on the panel (`suppressHeaderClick`). Text boxes and links are not
handles. The header's space after the buttons is at least 3.5 em wide
(`.forge-assistant-grip`), and the ✕ gone gives it that much more.

### The intelligent drag

A dragged box is drawn inside the room the anchors and Free Float place
against -- the window inside the safe-area insets and the gap (`dragRoom`) --
by `fitBox`, which is pure: a side the hand takes past an edge is held at that
edge and the box is cut on that side; across the left its left is held and its
right follows the hand; the top is never cut but stopped, the header being
there; nothing is cut below `least` (320 wide, 120 tall), past which the box
stops at the edge. The launcher is only stopped.

`fitDrag` applies it to the open panel on every move: its width written when
it changes, its height as a max-height written only where it binds and put
back once it does not (a max-height written on every move is a layout on every
move), its place still a transform. The panel's own size is read once, at the
drag's first move -- its width without a cut, and, for a panel an earlier drag
cut, its height measured at the window's cap -- so a cut panel grows back as it
is drawn in, and one taken again near the bottom is not taken to be as short as
it was cut. What is written on it is read rather than assumed. A panel as wide
as the room (a phone's, floating) is pinned across, as the sheet is: a finger
drifting sideways on its way down would otherwise narrow it for good. While the
panel is cut, a docked column is placed over its slot on each move
(`syncDock`) rather than carried by the transform, the slot having changed
size. And nothing places the panel under a drag that has moved (`placeNow`):
cutting its height changes its size, its ResizeObserver asked for a placement,
and the placement wrote the window's cap back over the drag's -- found by the
browser test, which saw a panel 156 px tall under a cap of 120.

Let go with Free Float on, the panel stays where and as it was drawn
(`endDrag`): what was cut is kept (`fitWidth`, `fitHeight`, saved with the
layout) and its place is the fraction of that cut box -- across, against the
edge; up and down, the header's own share of its travel (`headerFraction`),
never "against the bottom", which would mean its own height grown back up the
window. `placePanel` reads the two: a floating panel's width is the cut one,
its max-height no more than the cut one. Anchored, the release snaps to the
nearest anchor at the panel's own size, and the cut is forgotten. So is it when
the panel is put at a resting place (`putAway`; since §3.44 a panel put away
keeps its place and its cut), when Free Float is turned off,
and on the corner's reset; the corner, the edge and their keys take it as the
panel's own size first (`adoptFit`: the cut width as `panelWidth`, the
conversation as tall as it is drawn), so nothing springs back under the hand.

A panel let go against the bottom without being cut is still "against the
bottom" and grows upward (§3.30); a drag past the bottom now cuts it instead
of letting it off the window to be put back.

### Verified

`tests/test_assistant_drag_fit_js.py` (node): `fitBox` inside, across each
edge, at its least, the launcher stopped and a small box never enlarged;
`fitDrag` cutting at the right and the bottom and growing back; the release
keeping a cut (the header's fraction), forgetting it when dragged in and when
anchored; the placement keeping it floating and ignoring it anchored; what
forgets it and what adopts it (the corner, the edge, their keys); a press on a
header button uncaptured, uncancelled and uncounted until it moves; the ✕ and
the switch; Escape's order; a phone-wide panel pinned across.
`tests/test_assistant_taps_js.py`: one press on the header puts the panel
away (translated: it was nothing). `tests/test_assistant_layout.py`
(Chromium): a floating panel dragged past the right and then the bottom, held
at both edges and cut, let go with no bounce, taken again where it lies still
inside the window, dragged back in and grown back under the hand; past the
left its right follows the hand and past the top it stops, uncut; anchored it
stays on the window and snaps to its corner at its own size; cut at the bottom
a docked column stays over its slot; a drag from the Chat button moves the
panel and presses nothing while a tap on it is Chat; one tap on the header's
space puts the panel away after the window and leaves the launcher the focus,
Escape in the composer does with no ✕, and with the ✕ back it does not; the
handle's least width; and on a phone a finger dragging the floating panel past
the bottom, measured while the finger is down and after it lifts. Translated:
the transform and docked-column drags in node and Chromium now move inside the
room (their panels started against an edge, and a move past one now cuts the
panel), the bottom test lets go against the bottom rather than past it, the
docked column test closes the panel by one tap on the header, and the theme's
first-row test turns the switch off to see the ✕. Thirty-one mutations each
fail a test (two survived a first pass and the tests were tightened). Not run
on the user's machine or in a Gecko build.

## 3.44 The launcher and the panel, each at its own place (2026-10-28)

Asked for once §3.42 and §3.43 had merged (#307):

> "Let's change the global behavior of the open and closed positions of the
> flyout menu. I want the flyout menu open state to stay put when opened and
> closed, as well as the button to stay in its position. They should keep
> their positions independently. So, for example, if I have the button in the
> upper right, and I have the open state in the lower left, closing returns
> the button in the upper right. So it should keep the same position
> separately, independently. This should be the new global behavior for open
> and closing. The last PR wrote behavior I no longer want."

Since round one the launcher and the open panel had one place between them:
one of the six (`anchorOverride`) or, with Free Float on, one free place
(`floatAt`), written by a drag of either and read by `placePanel` for
whichever was on screen. §3.42 added a step on closing that moved that place
to the resting place nearest the panel. Both are gone.

### Two places

The launcher keeps the two names it had, so a saved place carries over; the
panel has two of its own, `panelAnchor` and `panelAt`. `placeNames(node)` says
which pair is a node's -- the panel's, or the launcher's; without a node, the
one on screen -- and `anchor(node)` and `floatOf(node)` read through it, the
anchor falling back to the default setting for each alike. Everything that
reads or writes a place goes through them:

- `placePanel` puts the node on screen at its own: anchored at its anchor,
  floating at its free place, the corner's held edges written into the
  panel's own. The root's `data-anchor` is the shown node's, so the phone's
  sheet takes its half from the panel's anchor.
- A drag starts from the dragged node's anchor (`startDrag`) and its free
  place (`moveDrag`), and `endDrag` writes only the dragged node's pair. A
  panel's cut (§3.43) stays the panel's, as before.
- `putAway` (a tap outside, one tap on the header) and the ✕ and Escape only
  put the panel away (`showOpen(false)`): the launcher shows at its own place,
  and the panel keeps its own and its cut for the next time it opens.
- `cornerSides` reads the panel's anchor.

### Free Float, each where it is

Turning Free Float on or off carries both, each where it is (`keepPlace`).
Turned on, the node on screen is measured and its free place is where it is
drawn (the old code did this for the shown node, and only when it had no free
place yet, so a stale one from an earlier turn moved it); the node put away
takes the free place exactly on its anchor (`anchorFraction`, which is exact
for the panel too, a free place of 1 down being against the bottom). Turned
off, the node on screen goes to the anchor nearest its box (`nearestAnchor`,
as a drag would choose), and the node put away to the anchor nearest its free
place (`anchorNear`: the six are the free places at none, half and all of the
way across and at either end up and down, so the nearest is the nearest of
those on each axis). Free places are kept for the next turn, as they were.

### From one place to two

A layout saved before this has one place. On reading it, the panel takes the
launcher's: a session layout without `panelAnchor` gives the panel the
`anchorOverride` it has (`_restore`), and a Free Float record without
`panelAt` gives the panel its `floatAt` (`_restoreFloat`), so nothing moves on
the update. A layout saved since says the panel's own, `null` when it has none,
and a panel with none stays at the default rather than taking the launcher's.
The session payload keeps schema 5: it gained a key, and a bump would have
thrown away the rest of the layout.

### Verified

`tests/test_assistant_places_js.py` (node, new): each node placed at its own,
anchored and floating, and the sheet's half the panel's; a drag of the
launcher writing the launcher's place alone, anchored and floating; Free Float
on and off carrying each where it is, the one on screen measured and the one
put away by its own place, a stale free place not used; a layout from before
giving the panel the shared place, one saved since coming back as two, and a
panel saved without a place staying without one.
`tests/test_assistant_tap_close_js.py`: putting the panel away by a tap
outside and by the header moves neither place nor the cut, anchored and
floating (replacing the test of the nearest of the six).
`tests/test_assistant_layout.py` (Chromium, new): the user's example by hand
-- the launcher dragged to the top right, the panel to the bottom left -- with
Free Float off and on, then a tap outside, one tap on the header, Escape and
the ✕, each showing the launcher where it was put and each opening the panel
where it was put (replacing the test of the nearest of the six). Translated:
every test that sets or reads an open panel's place sets or reads the panel's
own (`panelAt`, `panelAnchor`) -- in `test_assistant_js.py`'s Free Float
tests (with the launcher's place asserted untouched by a panel's drag and its
own when Free Float is turned on), the drag-fit, resize and header-tap tests,
and in Chromium the shared floating page, the corner, the phone's free float
and the gallery under a panel at the top -- and the cut is kept when the panel
is put away.
Twenty-one mutations each fail a test (two survived a first pass, one of them
a mutant the corner test's stub could not tell apart, and the tests and the
mutant were tightened). Not run on the user's machine or in a Gecko build.

## 3.45 The first tap outside only puts the panel away (2026-10-28)

Asked for on the same branch as §3.44, to go in the same pull request:

> "I want the first tap outside of an open flyout menu to do nothing except
> close the flyout menu. Essentially, catch all the taps and make flyout menu
> minimize action the priority. This means I have to tap twice to do
> something outside of an open flyout menu. Drag scroll should still work
> though."

Until now (§3.42) the tap closed the panel and went on to land: a button under
it was pressed, a text box under it took the keyboard.

### Taken at the press

Whether a press is a tap is known only when it is let go, and by then its
pointerdown has done its work -- a text box focused, a dropdown opened, a
slider moved, a stroke begun. So the press outside is taken from the page as
it goes down (`notePressOutside`, `holdBack`): cancelled, which starts no
focus, no selection, no native drag and no emulated mouse events, and stopped
on the window in the capture phase, so nothing on the page hears it. The tap
is now decided on the release (`tapOutside` on `pointerup`, no longer on the
click), the release taken too, and the panel put away there. A key between
the press and the release spends the press rather than forgetting it, so the
release is still taken (`spent`).

### What follows the release

A finger's tap goes on after `pointerup`: `touchend`, then the emulated
`mousedown` and `mouseup`, then the click; a mouse's sends `mouseup` and the
click. All of it is the press's (`holdTrail`, `outsideTrail`): held for
TAP_TRAIL_MS (1 s) after the release, in the press's own document, ending at
the click or at the next press. `touchend` is heard not passive and cancelled,
which sends no emulated mouse events and no click at all. While the press is
down, its `mousedown`, a long press's `contextmenu` and `selectstart` are held
as well. Only trusted events are.

### What is left to the page

Nothing that decides a scroll. A finger's pan is the browser's whatever the
pointerdown says (cancelling a pointerdown stops emulated mouse events, never
a pan), `touchstart` and `touchmove` are not listened to at all -- a
non-passive one would make the browser wait for the page before every pan --
a press the browser takes for a scroll is cancelled by it (`pointercancel`),
which ends the press with nothing held after it, and its `touchend` reaches
the page: the theme's momentum break reads a fling from its `touchstart` and
`touchend`. The wheel is never
touched, and a press on a scroll bar is not taken (`onScrollbar`).

A drag that is no scroll -- a slider, a selection, a stroke on a canvas -- is
the press's, taken with it, and does nothing while the panel is open, which
the press closing nothing leaves as it was. With one exception: a press on
something the browser drags of its own -- a picture, a link, anything
`draggable` (`dragsNatively`) -- is stopped and not cancelled, its
`mousedown` too, because cancelling it stops the drag, and a gallery picture
dragged onto the settings column docked in the panel (§3.41) is a drag the
panel is open for. Found by that feature's browser test, which failed with
every press cancelled. Its release and click are taken whole, so a tap on it
still only puts the panel away.

The keyboard's focus, which the tap no longer moves, goes to the launcher if
it was in the panel (`closeFromOutside`), as the ✕ and one tap on the header
give it; elsewhere it stays.

### Verified

`tests/test_assistant_tap_close_js.py` (node): a tap's pointerdown, its
mousedown, its release, its touchend, its mouseup and its click each
cancelled and stopped, the panel open until the release, a click after the
trail and the next tap's events passing; a finger's scroll (cancelled) leaving
its touchend and click to the page, a scroll bar's press, a press with the
panel closed or the switch off and the page's own events not taken, a drag
closing nothing but its click held, the trail ending after its time, at a new
press, and in its own document only; a press on a picture stopped but not
cancelled; focus to the launcher only from the panel; the listeners' set and
their options (`touchend` not passive, no `touchstart` or `touchmove`).
Translated: the wiring test's release is a `pointerup`, and a key spends the
press instead of forgetting it. `tests/test_assistant_layout.py` (Chromium):
a mouse's first tap on a page button closes the panel, presses nothing and is
heard by nothing on the page, the second presses it; a tap on a text box with
the composer focused gives the box nothing and the launcher the focus, the
second focuses the box; the wheel still scrolls the box it is over with the
panel open; on a phone a finger's scroll still scrolls the page, a finger's
first tap on a button presses nothing and the page hears no press, click or
emulated mouse event, the second presses it, and a tapped text box is not
given the keyboard; a tap in a same-origin frame does not press the frame's
button, the second does (translated from the tap that landed). The gallery
picture dragged onto the docked canvas holds the exception. Twenty-two
mutations each fail a test. Not run on the user's machine, in a Gecko build,
or against the real WanGP frame.

## 3.46 A drag of the launcher or the panel shows the hidden tab bar (2026-10-28)

Asked for once #308 had merged:

> "I have a function that shows or hides the tab bar when I'm in focus mode.
> And currently tapping the flyout will cause the tap bar to show up
> temporarily. Let's make that temporary show also map to moving / dragging
> the flyout menu or button. So simply jiggling the position of it will cause
> the tab bar to temporarily show up, and then it will dismiss itself."

A drag never reached the peek (§3.37): the peek is asked by `tapLauncher` and
`tapHeader`, and a drag's click is swallowed (`suppressClick`,
`suppressHeaderClick`), so no press of a drag was ever counted. Now:

- **At the drag's first move** -- the move past DRAG_THRESHOLD that makes a
  press a drag, in `moveDrag`'s once-a-drag block -- `peekBar` is asked, as a
  tap asks it: only in focus mode with the bar hidden and kept for this
  workspace, never in Super Focused. Not at the press: a press that does not
  move is a tap, which peeks already, and a press under the threshold is
  nothing new.
- **While the drag lasts** the bar is held (`peekHeld` answers yes for a drag
  that has moved), so a drag longer than PEEK_MS does not lose the bar under
  the hand.
- **At the release** its moment starts again (`endDrag`, `armPeek` when a
  drag that moved leaves a peek running), so the bar goes PEEK_MS after the
  hand lets go -- "and then it will dismiss itself".

Nothing else changes: the drag is still no press, so the panel neither opens
nor closes for it, and the setting's sentence says a tap or a drag.

### Verified

`tests/test_assistant_js.py` (node, `TestTheTabBarInFocusMode`): a mouse drag
of the launcher and of the panel peeks at its first move and not at the press
or a nudge under the threshold, and nothing with the bar shown or out of focus
mode; a drag held past the moment keeps the bar, the release arms a fresh
moment, and its end hides the bar. `tests/test_theme_layout.py` (Chromium,
`TestTheBarInFocusMode`): a mouse jiggling the launcher brings the bar over
the workspace, the bar stays through a hold longer than the moment and goes
the moment after the release, and the panel stays closed; a drag of the
panel's header brings it and leaves the panel open. Four mutations each fail a
test. Not run on the user's machine or in a Gecko build.

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

**Message editing in the flyout is the composer in an editing state** (§3.21).
The specification's in-panel edit buffer with Save/Cancel (§13.4) is not a
second box: the panel's own input takes the message, says it is an edit and
turns Send into Save. It carries the same (index, version, revision) as before
and is refused the same way. It replaced the browser's own prompt, which the
user rejected on sight.

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
