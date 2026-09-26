"""What actually goes on the wire for one chat turn.

The character's context becomes a system message, the history becomes the
turns after it, and the whole thing is trimmed to fit the context window
llama-server was started with. Three things are worth knowing about how:

*Placeholders are substituted, not passed through.* ``{{char}}``, ``{{user}}``
and the older ``<BOT>``/``<USER>`` appear throughout characters written for
other front ends, and a character telling the model to address someone called
"{{user}}" is a character that has been imported and not read.

*The persona is part of the system message, not a preamble.* An undefined
persona says nothing at all — the model is answering an unnamed "You", which is
what a chat with no persona has always been — and a defined one names you and
describes you where the character's own description is, because that is where
the model is looking when it decides who it is talking to.

*Trimming drops from the front, never the system message.* The character
survives a long conversation; the beginning of the conversation does not. The
budget is measured in characters against the context size in the setup state,
which is approximate on purpose: the exact figure is the model's tokenizer's to
know, and being wrong by a few hundred tokens costs nothing where being wrong
by thousands would.

*The front moves in steps, not every turn.* llama-server keeps the previous
prompt's key/value state and resumes the next prompt at their common prefix,
so a turn costs what is new at the end -- unless something near the start
changed, in which case everything after it is read again. Two things here used
to change the start every turn once a conversation was long enough: the window
dropped exactly one message from the front per turn, and each new picture took
the still off the oldest picture message, rewriting it. From one user's
llama-server log, on an Intel GPU reading 25 to 50 tokens a second, that was a
prompt of five thousand tokens read from the first token every turn -- three
minutes of "Replying…" before a word appeared, with only 92 or 335 tokens
found in the cache. So the front now moves in steps of a quarter of the
window rather than by one message a turn, and only the newest picture is
carried as a still: a trim changes the front rarely, and a new picture
rewrites only the previous picture's message, which is recent.
"""

from __future__ import annotations

from typing import Any

from .characters import Character, Persona
from .history import ASSISTANT, USER, Message

# Rough characters per token for English prose. Deliberately pessimistic — an
# overestimate of the tokens a message costs trims one message too many, where
# an underestimate overflows the window and truncates the character.
CHARS_PER_TOKEN = 3.2

# Once the history no longer fits, the front of the window moves in steps of
# this fraction of the budget rather than by one message every turn. The step
# is measured in cost from the beginning of the conversation, which is where
# the boundaries have to be for them to stay put as messages are appended;
# the front rests on a boundary until the newest messages that fit have
# walked past it, then jumps to the next. Every move of the front throws away
# llama.cpp's cached prefix, so a quarter of the window is one whole read of
# the prompt every several turns instead of every turn, at the price of a
# window that is at worst a quarter smaller right after a move.
TRIM_STEP = 0.25

# The model is shown the newest picture and no other, unless asked to see
# every one. Every older picture message keeps its text and says that an
# image was there -- and stays a picture in the chat, which draws from the
# attachment and not from what the prompt carried. One still is what a
# conversation is usually about, and it is what keeps the prompt's prefix: the
# only message a new picture rewrites is the previous picture's, which is
# recent, so everything before it stays in llama.cpp's cache. A still taken
# off an *old* message, which the earlier rule did on every new picture, threw
# the whole conversation out of it. With every picture shown, nothing is ever
# taken off a message, so the prefix holds too; what it costs is the window,
# at IMAGE_TOKENS a still.
MAX_IMAGES = 1
IMAGE_NOTE = "[image: {name}]"
IMAGE_TOKENS = 300


def stills_carried(pictures: int, every: bool = False) -> int:
    """How many of the newest ``pictures`` picture messages keep their still.

    :data:`MAX_IMAGES` of them, which is one: the newest -- or every one of
    them when ``every`` is set, which is the *Show the model every picture*
    setting. Shared with the reader that loads the stills off disk, so the two
    never disagree about which picture goes -- a builder dropping one the
    reader had loaded would rewrite a message the cache had, for nothing.
    """
    count = max(int(pictures), 0)
    return count if every else min(count, MAX_IMAGES)


def substitute(text: str, character: str, user: str) -> str:
    """The placeholder spellings every character format uses."""
    for token in ("{{char}}", "{{Char}}", "{{CHAR}}", "<BOT>", "<bot>"):
        text = text.replace(token, character)
    for token in ("{{user}}", "{{User}}", "{{USER}}", "<USER>", "<user>"):
        text = text.replace(token, user)
    return text


def system_text(character: Character, persona: Persona) -> str:
    """The system message: who the model is, and who it is talking to."""
    name = character.name.strip() or "the character"
    you = persona.display
    if character.system.strip():
        # A custom system message replaces the wrapper entirely, and is the one
        # place a character can say something this application would not.
        return substitute(character.system.strip(), name, you)

    parts = [f"You are {name}. Stay in character as {name} for the whole conversation.",
             "Write only your own reply — never the other person's lines, and never a "
             "name or label before it."]
    context = substitute(character.context.strip(), name, you)
    if context:
        parts.append(f"About {name}:\n{context}")
    if persona.description.strip():
        parts.append(f"You are talking to {you}. About {you}:\n"
                     f"{substitute(persona.description.strip(), name, you)}")
    elif persona.name.strip():
        parts.append(f"You are talking to {you}.")
    return "\n\n".join(parts)


def greeting_text(character: Character, persona: Persona) -> str:
    return substitute(character.greeting.strip(), character.name.strip() or "the character",
                      persona.display)


def build(character: Character, persona: Persona, messages: list[Message],
          context_size: int = 8192, reply_tokens: int = 512,
          instruction: str | None = None, every_picture: bool = False) -> list[dict[str, Any]]:
    """The request body's ``messages``, trimmed to fit.

    ``instruction`` is appended as a final user turn when the caller wants
    something other than the next reply — continuing the last one, or writing a
    message as the user. It is never stored in the history.

    ``every_picture`` sends every picture in the window as a still rather than
    the newest alone; see :func:`stills_carried`.
    """
    system = system_text(character, persona)
    budget = _budget(context_size, reply_tokens, system, instruction or "")
    kept = _fit(messages, budget, every_picture)
    with_images = _limit_images(kept, every_picture)
    wire: list[dict[str, Any]] = [{"role": "system", "content": system}]
    name, you = character.name.strip() or "the character", persona.display
    for message, keep_image in with_images:
        text = substitute(message.text.strip(), name, you)
        if not text and not keep_image:
            continue
        wire.append({"role": message.role, "content": _content(message, text, keep_image)})
    if instruction:
        wire.append({"role": USER, "content": instruction})
    return wire


def continue_instruction(character: Character) -> str:
    """Ask for more of the reply that is already on screen.

    llama-server's chat endpoint closes the assistant turn it is given, so a
    continuation is asked for rather than prefilled — and the caller appends
    what comes back to the text already there.
    """
    name = character.name.strip() or "the character"
    return (f"Continue {name}'s last message from exactly where it stopped. Do not repeat "
            "any part of it, do not start it again, and do not add any preamble — write only "
            "the text that follows on.")


def prefix_instruction(character: Character) -> str:
    """Ask for the rest of a reply whose opening was written by hand.

    The opening is already on the wire as the last assistant turn, put there by
    the caller — which is also what makes the reply *start* with it, rather than
    the model being asked nicely and mostly complying. So this says two things a
    plain continuation does not need to: carry on from it, and keep to it. A
    start the character would never have chosen is the entire point of writing
    one, and a model told only to "continue" will happily talk its way back out
    of a premise it disagrees with.
    """
    name = character.name.strip() or "the character"
    return (f"{name}'s reply has already been started for you: the last message above is the "
            f"opening of it, and it is fixed. Carry straight on from its final character in "
            f"{name}'s voice, writing as though {name} had chosen those words — accept whatever "
            "that opening asserts or commits to, even where you would have said something else, "
            "and follow it through. Do not repeat it, do not begin the reply again, do not "
            "contradict or walk it back, and do not add any preamble or commentary — write only "
            "the text that follows on from it.")


def impersonate_instruction(persona: Persona) -> str:
    """Ask the model to write your next message instead of the character's."""
    about = f" About them: {persona.description.strip()}" if persona.description.strip() else ""
    return (f"Write the next message as {persona.display} — the person you are talking to — "
            f"in their own voice and in the first person.{about} Write only that message, with "
            "no name label and no commentary.")


def clean_reply(text: str, character: Character, persona: Persona) -> str:
    """Strip a name label the model wrote anyway.

    Models trained on chat transcripts open with "Name:" often enough that the
    system message asking them not to is not sufficient on its own.
    """
    reply = text.strip()
    for name in (character.name.strip(), persona.display, "Assistant"):
        if name and reply.casefold().startswith(f"{name.casefold()}:"):
            reply = reply[len(name) + 1:].lstrip()
    return reply


def _content(message: Message, text: str, keep_image: bool) -> Any:
    """The multimodal shape llama-server takes: the image part, then the text."""
    if keep_image and message.image:
        return [{"type": "image_url", "image_url": {"url": message.image}},
                {"type": "text", "text": text or "(no text)"}]
    if message.image and not keep_image:
        note = IMAGE_NOTE.format(name=message.image_name or "attached")
        return f"{note}\n{text}".strip()
    return text


def _budget(context_size: int, reply_tokens: int, system: str, instruction: str) -> int:
    """Characters of history the window has room for."""
    spare = int(context_size) - int(reply_tokens) - _tokens(system) - _tokens(instruction)
    # A floor rather than a negative budget: a context this small cannot hold
    # the character either, and the last message is always sent.
    return max(400, int(spare * CHARS_PER_TOKEN))


def _tokens(text: str) -> int:
    return int(len(text) / CHARS_PER_TOKEN) + 8


def _cost(message: Message, still: bool = False) -> int:
    """What a message costs the window, in characters. A still is charged
    :data:`IMAGE_TOKENS`; a picture carried as a note costs its note."""
    extra = 0
    if message.image:
        extra = int(IMAGE_TOKENS * CHARS_PER_TOKEN) if still \
            else len(IMAGE_NOTE.format(name=message.image_name or "attached")) + 1
    return len(message.text) + 32 + extra


def _fit(messages: list[Message], budget: int, every: bool = False) -> list[Message]:
    """The newest messages that fit, oldest-first. The last one always does.

    A history that fits is sent whole. One that does not is cut at a front
    that moves in steps of :data:`TRIM_STEP` of the budget rather than by one
    message a turn -- see the module docstring for what a moving front costs.
    Only the pictures sent as stills are charged as stills -- the newest, or
    every one when ``every`` is set (:func:`_limit_images`).
    """
    pictured = [index for index, message in enumerate(messages) if message.image]
    stills = set(pictured[len(pictured) - stills_carried(len(pictured), every):])
    costs = [_cost(message, still=index in stills)
             for index, message in enumerate(messages)]
    if sum(costs) <= budget:
        return list(messages)
    # The newest messages that fit, the last of them whatever it costs.
    front = len(messages) - 1
    spent = costs[front]
    while front > 0 and spent + costs[front - 1] <= budget:
        front -= 1
        spent += costs[front]
    # Then on to the next boundary. Boundaries are multiples of the step in
    # the cost of everything before the front, pictures counted as notes so
    # that a picture growing old behind the front cannot move them.
    step = int(budget * TRIM_STEP)
    if step > 0:
        behind = sum(_cost(message) for message in messages[:front])
        boundary = -(-behind // step) * step
        while front < len(messages) - 1 and behind < boundary:
            behind += _cost(messages[front])
            front += 1
    return list(messages[front:])


def _limit_images(messages: list[Message], every: bool = False) -> list[tuple[Message, bool]]:
    """Which of the kept messages still carry their still. See :func:`stills_carried`."""
    allowance = stills_carried(sum(1 for message in messages if message.image), every)
    marked = []
    for message in reversed(messages):
        keep = bool(message.image) and allowance > 0
        allowance -= 1 if keep else 0
        marked.append((message, keep))
    marked.reverse()
    return marked


def has_image(messages: list[Message]) -> bool:
    """Whether any of these conversation messages carries a still."""
    return any(message.image for message in messages)


def needs_vision(wire: list[dict[str, Any]]) -> bool:
    """Whether the request :func:`build` produced actually carries an image.

    Asked of the built payload rather than of the history it was built from,
    because those two can disagree and only one of them is what llama-server
    will be shown. A long conversation trims to the context it has room for and
    then keeps at most :data:`MAX_IMAGES` of the stills that survived, so a chat
    whose only picture scrolled out of the window is a text-only request -- and
    answering "yes, vision" about it would pay for a projector upgrade to send a
    request with nothing for the projector to look at.

    The other direction cannot happen: an image part in the payload came from a
    message that had one. That asymmetry is the point. This never under-reports,
    so invariant I-1 -- an existing feature that needs image understanding still
    asks for it -- holds by construction.
    """
    for message in wire or ():
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, (list, tuple)):
            continue
        if any(isinstance(part, dict) and part.get("type") == "image_url" for part in content):
            return True
    return False


__all__ = ["ASSISTANT", "USER", "build", "clean_reply", "continue_instruction",
           "greeting_text", "has_image", "impersonate_instruction", "needs_vision",
           "prefix_instruction", "stills_carried", "substitute", "system_text"]
