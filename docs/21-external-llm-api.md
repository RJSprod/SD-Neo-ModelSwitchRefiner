# External LLM API — MiniMax H3 prompts, V1

**For developers of other Forge / SD WebUI Neo extensions.**

This document is the contract. `mc_llm_api` is the module you import,
`tests/test_llm_external_api.py` is the executable version of everything below,
and if the two ever disagree the tests are right and this file is a bug.

**Scope: MiniMax H3 prompt enhancement only.** Prompt Studio, Conversation and
Krea 2 have no external surface and are not planned for V1.


## 1. What it is, in one sentence

Your extension asks Model Chain for a MiniMax H3 prompt exactly as if somebody
had gone to **LLM Studio → MiniMax H3** and pressed *Write H3 Prompt* — except
that requests are queued, carry an id, and can be tracked and cancelled from
code.

Everything underneath is the machinery that was already there. Your request
runs the same `mc_llm_sessions.minimax` generator, over the same
`llama-server`, under the same GPU workload lock, with the same vendored WanGP
instructions, and its result is filed in the same *Saved prompts* history. It
waits for an image generation on the same card for the same reasons a panel run
does.

Three things are new, and they are all consequences of nobody watching a
screen:

| | Panel | Your extension |
|---|---|---|
| Concurrency | a person cannot press Enhance twice at once | requests are queued, first in first out |
| Identity | the answer is on screen | the answer is collected by id |
| Instructions | `@@` in the prompt text | a `system_prompt` parameter |


## 2. It is an import, not a URL

```python
import mc_llm_api
```

There is no HTTP route, no port, no token and no account. Your extension is
Python running inside the same Forge process as this one, so the honest
transport between the two is a function call. An HTTP layer would have meant
base64-ing a `PIL.Image` you already hold, posting it to `localhost` so this
process could decode it back into the object it started as, and inventing a
credential for a caller that shares our address space.

**If you need HTTP, you already have somewhere to put it.** Events come back as
plain dicts shaped like Server-Sent Events (`seq`, `event`, `data`), so
forwarding a request to your own browser panel is a handful of lines on *your*
routes, under *your* auth:

```python
import json

from starlette.responses import StreamingResponse

def stream(job_id: str):
    feed = mc_llm_api.subscribe(job_id)

    def lines():
        for event in feed.events():
            yield f"id: {event['seq']}\nevent: {event['event']}\n" \
                  f"data: {json.dumps(event['data'])}\n\n"

    return StreamingResponse(lines(), media_type="text/event-stream")
```

Note what that gives you for free: `event['seq']` becomes the SSE `id:` field,
so a browser's `EventSource` sends `Last-Event-ID` on reconnect and you pass it
straight back as `cursor=`.

`mc_llm_api` performs no authorisation of its own. Anything that can import it
is already running with the WebUI's privileges. `origin` is a label for the
console and for the panel's banner — it is not a claim anybody checks.


## 3. Getting a prompt written

```python
job_id = mc_llm_api.submit_minimax(
    "a car chase through a rainy city at dusk",
    variant="fl2va",
    first_frame=opening_frame,          # optional
    last_frame=closing_frame,           # optional
    reference=None,                     # optional
    system_prompt=None,                 # optional override
    seed=None,                          # optional; drawn per request
    origin="my-extension",              # optional label
    remember=True,                      # file it in Saved prompts
)
```

Returns a string id immediately. Nothing has run yet.

| Argument | Required | Notes |
|---|---|---|
| `prompt` | **yes** | Must not be blank. `@` and `@@` mean what they mean in WanGP — see §6. |
| `variant` | no | `"fl2va"` or `"ref2va"`, case-insensitive. Anything else, including the default, resolves to `fl2va`. |
| `first_frame`, `last_frame`, `reference` | no | Any combination. See §4. |
| `system_prompt` | no | Replaces this variant's instructions for this request. Blank is refused. |
| `seed` | no | `None` draws a fresh one per request, matching the panel and WanGP's `prompt_enhancer_randomize_seed`. Pass an `int` to reproduce a run. |
| `origin` | no | Free text, trimmed to 120 characters. Shown in the console and in the panel's banner. |
| `remember` | no | `True` files the result in LLM Studio's *Saved prompts*. Pass `False` for bulk work you do not want in somebody's history. |

`submit_minimax` raises `mc_llm_api.Rejected` — and nothing else — for anything
you can act on. It carries a `.code`:

| `.code` | Means |
|---|---|
| `disabled` | LLM Studio is switched off in this WebUI's settings. The tab does not exist, so neither does this. |
| `empty_prompt` | The prompt was blank. |
| `empty_system_prompt` | An override was given but was blank. Leave it out to use the default. |
| `bad_image` | One of the pictures could not be read. The message names which slot. |
| `no_vision` | A picture was sent but the model running has no vision projector. Nothing was queued. The same refusal the panel makes before starting. |
| `queue_full` | `MAX_QUEUED` (32) requests are already waiting. Retry later. |
| `empty_origin` | `cancel_all()` was called without an origin (§9). |
| `unknown_job` | `subscribe()` was given an id that does not exist (§7). |

New codes may be added; treat one you do not recognise as a refusal you cannot
retry your way past, and show its message.


## 4. Pictures: three slots, one caption

Each of `first_frame`, `last_frame` and `reference` accepts any of:

* a `PIL.Image`
* a `pathlib.Path` or a path string
* raw image bytes (PNG, JPEG or WebP)
* an existing `data:` URL, passed through untouched

**Exactly one of them is captioned.** This is not a limitation of the queue; it
is what the vendored path does. WanGP runs one caption pass and hands the
enhancer a single `image_caption:` line, and inventing a second line here would
be inventing a prompt format the H3 models were not trained against.

Which one is chosen:

| Variant | Preference order |
|---|---|
| `fl2va` | `first_frame` → `last_frame` → `reference` |
| `ref2va` | `reference` → `first_frame` → `last_frame` |

The fallbacks matter: if you ask for `fl2va` and supply only a `reference`, the
prompt is written about your reference rather than written as text-only with
your picture silently discarded.

You never have to guess which one was used. The job record says so:

```python
found = mc_llm_api.status(job_id)
found["images"]         # ["first_frame", "last_frame"] — what you supplied
found["image_used"]     # "first_frame"                 — what was captioned
found["image_ignored"]  # ["last_frame"]                — what was not
```

Any picture at all also switches the enhancer from its text instructions to its
image instructions, which is WanGP's behaviour and not something the slot
choice changes.

**Vision is required for any picture.** If the model running has no vision
projector, `submit_minimax` raises `Rejected("no_vision")` and queues nothing —
the same refusal the panel makes before it starts. `capabilities()["vision"]`
(§8) tells you in advance so you can say so in your own UI. If that
configuration cannot be read at all, the request is queued and the run itself
decides, failing with the vendored client's own message if it must.

**No picture bytes are kept.** The data URL exists for as long as the run needs
it and is dropped when the run ends. It never appears in `status()`, in a log
line, or in the saved history — which records a name only: a file's basename
(never its path), or, for a picture that arrived with no name, the slot it
filled (`"first_frame"`).


## 5. Tracking it

### `status(job_id) -> dict | None`

`None` means the record has been forgotten (§9), which is an ordinary answer
fifteen minutes after a request finished — not an error.

```python
{
  "id": "8e8983b7cd4f418d",
  "kind": "minimax",
  "state": "done",              # queued | running | done | failed | cancelled
  "origin": "my-extension",
  "variant": "fl2va",
  "seed": 1274091,
  "created": 1789012345.67,     # epoch seconds
  "started": 1789012346.01,     # None while queued
  "finished": 1789012389.42,    # None until terminal
  "elapsed": 43.41,             # seconds since it started (see below)
  "queued_for": 0.34,           # seconds spent in the line before that
  "position": 0,                # 1 == next to run; 0 == running or finished
  "cancelling": False,          # a stop has been asked for but not yet taken
  "stage": "Describing the image…",   # the last thing the run said it was doing
  "system_override": False,
  "images": ["first_frame"],
  "image_used": "first_frame",
  "image_ignored": [],
  "events": 47,                 # highest seq emitted so far
  "dropped_events": 0,          # chunk events trimmed from the log
  "prompt": "…the finished H3 prompt…",
  "caption": "…the image description it was written from…",
  "request": "…the prompt you sent…",
  # "error"  — present only when state == "failed"
  # "reason" — present only when state == "cancelled"
}
```

`result(job_id)` is shorthand for the finished prompt alone, and is `""` until
there is one.

**`stage` is what "running" is actually doing.** A request is `running` from
the moment the worker takes it off the line, and that includes time spent
waiting for the GPU behind an image generation or a panel run on the same card
— the same wait the panel shows as *Waiting for …*. `stage` carries that text
(`"Waiting for image generation on GPU 0…"`, `"Describing the image…"`, and
so on), so a caller that only polls can tell a request that is generating from
one that is queued behind something the queue cannot see. `elapsed` counts
from `started` and therefore includes that wait.

### `queue(limit=20, origin=None) -> dict`

One consistent read of the queue. Assembling this from separate calls can
describe a state that never existed, so it is a single function.

```python
{
  "active": True,
  "running": {...},        # a status dict without the prompt text, or None
  "waiting": 2,
  "queue": [{...}, {...}], # in the order they will run
  "recent": [{...}],       # most recently finished first
  "capacity": 32,
  "origin": None,          # the filter this listing was made under
}
```

`origin="my-extension"` narrows `running`, `queue` and `recent` to requests
submitted under that label — what your own panel wants to draw. `active` and
`waiting` stay the whole queue's on purpose: your next request is behind
everything in the line, not only behind your own, and that is the number you
would use to predict how long it takes.

`busy()` is the one-line version: `True` if anything external is running or
waiting.

Neither `queue()` nor `capabilities()` carries anybody's prompt text.


## 6. Overriding the instructions

The four defaults are readable, which is the other half of being able to
replace one honestly:

```python
mc_llm_api.system_prompts()
# {"fl2va": {"label": "FL2VA — from text or a frame",
#            "text":  "…instructions used with no picture…",
#            "image": "…instructions used with a picture…",
#            "structure": "…what an H3 prompt is made of…",
#            "max_tokens": 1024},
#  "ref2va": {...}}

mc_llm_api.system_prompt("ref2va", has_image=True)   # just one of them
```

There are **four**, not two: WanGP reads a different instruction set when the
generation has a picture than when it does not, on each of the two H3 model
definitions.

Passing `system_prompt=` replaces whichever of the four this request would have
used. WanGP's own prompt dialect still applies on top, so an override composes
rather than conflicting:

| In the prompt text | Effect |
|---|---|
| `body @ extra` | `extra` is appended to the instructions — your override, if you gave one |
| `body @@ replacement` | `replacement` replaces the instructions entirely, beating your override |

`status()["system_override"]` records whether a request ran under its defaults.
The override text itself is never logged, exactly as prompt text is never
logged.


## 7. Feeds

```python
feed = mc_llm_api.subscribe(job_id)          # one request
feed = mc_llm_api.subscribe()                # the whole queue
feed = mc_llm_api.subscribe(job_id, cursor=n, ttl=300.0)
```

`subscribe` raises `Rejected("unknown_job")` for an id that does not exist,
rather than handing back a feed that will never produce anything. The class it
returns is `mc_llm_api.Feed`, re-exported for annotations.

Two ways to read one:

```python
for event in feed.events():          # blocks, yields as things happen
    ...
for event in feed.events(timeout=30.0, idle=1.0):
    ...

batch = feed.poll()                  # everything since the cursor, right now
```

`idle` is how long a single internal wait may block — it is what lets a caller
forwarding this to an HTTP response notice its client has gone away. `timeout`
is the real limit; `None` means "until this feed closes", which for a
request-bound feed is a bounded promise because every request reaches a
terminal event. Nothing is yielded while the queue's lock is held, so a slow
consumer cannot stall the worker.

### Event shape

```python
{"seq": 12, "stream": 481, "id": "8e8983b7cd4f418d",
 "event": "chunk", "time": 1789012350.11, "data": {"text": "…"}}
```

| `event` | `data` | When |
|---|---|---|
| `queued` | `position`, `waiting`, `running` | Accepted into the line |
| `position` | `position`, `waiting` | The line shortened ahead of it |
| `started` | `variant`, `seed` | Taken off the queue; the run begins |
| `status` | `text` | A stage of the run — the same text the panel shows |
| `caption` | `text` | The picture has been described |
| `chunk` | `text` | A piece of the prompt, as it is written |
| `done` | `prompt`, `caption`, `seed`, `variant`, `elapsed` | Finished |
| `failed` | `error` | Did not finish |
| `cancelled` | `reason` | Stopped |

Exactly one of `done`, `failed` or `cancelled` ever arrives.

### Cursors

`seq` counts within one request; `stream` counts across the whole process. A
request-bound feed's `cursor` is a `seq`; a queue-wide feed's is a `stream`.
Pass the last one you handled and you get exactly what you missed — no
duplicates and no gap. `cursor=0`, the default, replays from `queued`, which
makes the gap between `submit_minimax` and `subscribe` harmless.

### Lifecycle

* A feed bound to one request **closes itself** after delivering that request's
  terminal event. There will never be another.
* A queue-wide feed has no such moment. It closes on `feed.close()` or on TTL.
* A feed nobody has read for `FEED_TTL` (300 s) is dropped. The clock is
  read-idle, so a feed being actively consumed never expires however long its
  request takes.
* Forgetting a request (§9) closes the feeds watching it.
* Call `feed.close()` when you are done. `mc_llm_jobs.feeds()` lists what is
  open if you want to check your own housekeeping.

### The event log is bounded

At most `MAX_EVENTS` (600) events are kept per request, and **only `chunk`
events are ever dropped** — every `status`, `caption` and terminal event
survives, and `done` carries the complete prompt. A subscriber that fell behind
loses some of the typewriter effect and none of the answer.
`status()["dropped_events"]` says how many.


## 8. Asking what is possible before asking for it

```python
mc_llm_api.capabilities()
# {"api_version": 1,
#  "kinds": ("minimax",),
#  "variants": ["fl2va", "ref2va"],
#  "slots": ["first_frame", "last_frame", "reference"],
#  "events": [...],
#  "max_queued": 32, "feed_ttl": 300.0, "job_retention": 900.0,
#  "enabled": True,           # LLM Studio is switched on at all
#  "configured": True,        # a language model is set up
#  "vision": True,            # …and it can see pictures
#  "model": "qwen3-vl-8b.gguf",
#  "reason": ""}              # why not, when any of the three is False
```

Every field is a fact about this machine at this moment, not a promise:
`vision` can go false when somebody switches models, and `enabled` when
somebody turns the tab off. Reading this first turns a refusal into a message
you can show before the user presses anything. `reason` names the first of the
three that is false, in the order a person would fix them.

`API_VERSION` is bumped when something documented here changes meaning. Adding
a field, an event name or a keyword argument does not bump it — so read fields
you know and ignore ones you do not.


## 9. Cancelling, and what a queue guarantees

```python
mc_llm_api.cancel(job_id, "the user closed the panel")
```

| Where it was | Result | Returned |
|---|---|---|
| Queued | Leaves the line. The card is never touched. | `{"ok": True, "state": "cancelled", "was": "queued"}` |
| Running | The same stop the panel's Stop button uses. | `{"ok": True, "state": "cancelling", "was": "running"}` |
| Finished | Refused. | `{"ok": False, "code": "already_finished", ...}` |
| Unknown | Refused. | `{"ok": False, "code": "unknown_job", ...}` |

```python
mc_llm_api.cancel_all("my-extension", "the user closed the panel")
```

cancels every request submitted under that origin, running or waiting, and
returns `{"ok": True, "cancelled": n, "ids": [...]}`. The origin is required —
"cancel everything, whoever asked for it" is the user's decision to make from
the panel, not something a caller should reach by forgetting an argument — and
a blank one is refused with `empty_origin`. (`mc_llm_jobs.cancel_all()` with no
origin exists for the panel and does take everyone down.)

`"cancelling"` is deliberate. `llama.cpp` honours a stop between tokens, so the
call returns before the run has actually ended. Watch the feed for the
`cancelled` event if you need to know it really stopped. Cancellation is
cooperative and leaves residency exactly as it found it — the server stays up
and the next request is a warm one.

**Nothing preempts anything.** Arriving does not jump the line, position cannot
be bought, and nothing cancels somebody else's request to make room. A request
that is running is never interrupted by one that arrives during it. This is the
property the whole feature was asked for.

The same holds in the other direction, through the lock rather than the queue:
a panel run — MiniMax, Krea, a conversation reply — that began before your
request holds the GPU workload lock, and your request waits for it exactly as
it waits for an image generation. It is `running` with a `stage` of *Waiting
for …* while that happens. It is never started on top of one.

**One at a time.** Two external requests never run concurrently, even on a
two-card machine. `llama-server` is one process per role and its prompt cache
is per process, so a second concurrent enhancement would be two requests taking
turns *inside* the server instead of outside it — no faster, and no longer
traceable.

### Retention

A finished request stays readable for `JOB_RETENTION` (900 s) or until there
are more than 200 of them, whichever comes first — long enough that a caller
which crashed and restarted can still collect a result it never saw. Call
`forget(job_id)` to drop one early; it returns `False` for a request that has
not finished. Nothing is written to disk: records live in this process's RAM
and do not survive a WebUI restart. The finished prompt does survive, in *Saved
prompts*, when `remember=True`.


## 10. What the user sees

While anything external is running or waiting, **LLM Studio → MiniMax H3 is
inert**: the Enhance button is disabled behind a banner naming who is asking,
how long the running request has taken, its id, and how many are behind it.

The banner offers two buttons:

* **Stop the running request** — cancels the one on the card, leaves the queue.
* **Cancel all queued** — cancels everything external.

So a person is never blocked without being told why or given a way out. This
matters for you in one direction: **your request can be cancelled by the user
at any time.** You will see a `cancelled` event with a `reason` naming LLM
Studio. Handle it as an ordinary outcome, not an error.

The gate is re-read when the workspace is opened, on page load, on *Check
again*, after every panel run ends, every two seconds while the workspace is
on screen (on hosts whose Gradio has `gr.Timer`), and — authoritatively — by
the Enhance handler itself at the moment it is pressed.

**If LLM Studio is switched off** (Settings → Model Chain), the tab does not
exist and every request is refused with `disabled`. The toggle turns off the
whole LLM half, this API included.

### What the console says

Every run this API starts writes the same console lines a panel run writes,
with the request named on the end of each:

```
Model Chain: MiniMax request 8e8983b7cd4f418d queued at position 1 for my-extension
Model Chain: LLM run started — a MiniMax fl2va enhancement (request 8e8983b7cd4f418d from my-extension)
Model Chain: LLM a MiniMax fl2va enhancement (request 8e8983b7cd4f418d from my-extension) — Describing the image…
Model Chain: LLM run finished — a MiniMax fl2va enhancement (request 8e8983b7cd4f418d from my-extension), 1,412 characters in 43.4s
Model Chain: MiniMax request 8e8983b7cd4f418d done after 43.4s
```

Nothing written there is content: no prompt, no override, no caption, no
picture. `origin` is the one caller-supplied string that reaches the log, which
is worth knowing when you choose it.


## 11. A complete example

```python
import json
import logging

import mc_llm_api

logger = logging.getLogger("my_extension")


def write_h3_prompt(rough: str, first_frame=None, last_frame=None):
    ready = mc_llm_api.capabilities()
    if not ready["configured"]:
        raise RuntimeError(ready["reason"])
    if (first_frame is not None or last_frame is not None) and not ready["vision"]:
        raise RuntimeError(ready["reason"])

    try:
        job_id = mc_llm_api.submit_minimax(
            rough, variant="fl2va",
            first_frame=first_frame, last_frame=last_frame,
            origin="my-extension")
    except mc_llm_api.Rejected as refused:
        logger.warning("Model Chain refused the request (%s): %s",
                       refused.code, refused.reason)
        raise

    feed = mc_llm_api.subscribe(job_id)
    try:
        for event in feed.events(timeout=600.0):
            if event["event"] == "position":
                logger.info("waiting, position %s", event["data"]["position"])
            elif event["event"] == "caption":
                logger.info("described the frame")
            elif event["event"] == "done":
                return event["data"]["prompt"]
            elif event["event"] == "failed":
                raise RuntimeError(event["data"]["error"])
            elif event["event"] == "cancelled":
                raise RuntimeError(f"cancelled: {event['data']['reason']}")
    finally:
        feed.close()

    # The feed timed out rather than ending. The request is still ours.
    mc_llm_api.cancel(job_id, "my-extension gave up waiting")
    raise TimeoutError(f"no answer for {job_id}")
```

Polling instead of subscribing is equally supported — `status(job_id)["state"]`
reaches a terminal value and stays there until retention drops the record.


## 12. Where the code is

| File | What |
|---|---|
| `mc_llm_api.py` | The surface you import. Nothing else here is API. |
| `mc_llm_jobs.py` | The queue, the records, the feeds, the worker. |
| `mc_llm_sessions.py` | The run itself, shared with the panel. |
| `mc_llm_minimax_panel.py` | The panel and its gate. |
| `prompt_master/minimax/` | The vendored WanGP instructions. |
| `tests/test_llm_external_api.py` | This document, executable. |

`mc_llm_jobs` is not private, and reading it is fine — `mc_llm_jobs.feeds()`
and the state constants are useful. But `mc_llm_api` is the part that carries
`API_VERSION`, and it is the part that will be kept working.


## 13. Revisions

Everything below is additive. `API_VERSION` is still `1`; a caller written
against the first revision keeps working unchanged, and one that matches on
refusal codes exhaustively should read the note under §3.

**V1, second revision** — after review against the original brief:

| Added | Where |
|---|---|
| `Rejected("disabled")` when LLM Studio is switched off | §3, §10 |
| `Rejected("no_vision")` at submit time for a picture the model cannot see, instead of a queued job that fails | §3, §4 |
| `status()["stage"]` — what a `running` request is actually doing, waits included | §5 |
| `queue(origin=…)` — a caller's own requests | §5 |
| `cancel_all(origin, reason)` — a caller's own requests, origin required | §9 |
| `capabilities()["enabled"]`; `kinds` is now a list | §8 |
| `mc_llm_api.Feed` re-exported | §7 |
| Saved-prompts history records a file's basename, as the panel does; a nameless picture is recorded by its slot | §4 |
| Every console line names the request and its origin | §10 |
| The panel re-reads its gate after every run of its own, so a host without `gr.Timer` never shows an enabled Enhance over a running external request; the panel's Stop does the same; the busy refusal no longer clears the last prompt off the screen | §10 |
| A status read is taken under the queue's lock, so a record is never seen half-written | §5 |

**V1, first revision** — the original surface: `submit_minimax`, `status`,
`result`, `queue`, `busy`, `cancel`, `forget`, `subscribe`, `system_prompt`,
`system_prompts`, `variants`, `capabilities`.
