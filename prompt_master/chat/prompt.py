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
"""

from __future__ import annotations

from typing import Any

from .characters import Character, Persona
from .history import ASSISTANT, USER, Message

# Rough characters per token for English prose. Deliberately pessimistic — an
# overestimate of the tokens a message costs trims one message too many, where
# an underestimate overflows the window and truncates the character.
CHARS_PER_TOKEN = 3.2

# Vision costs the same whether the still is three turns old or thirty, so only
# the most recent few are carried. Older messages keep their text and say that
# an image was there.
MAX_IMAGES = 4
IMAGE_NOTE = "[image: {name}]"
IMAGE_TOKENS = 300


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
          instruction: str | None = None) -> list[dict[str, Any]]:
    """The request body's ``messages``, trimmed to fit.

    ``instruction`` is appended as a final user turn when the caller wants
    something other than the next reply — continuing the last one, or writing a
    message as the user. It is never stored in the history.
    """
    system = system_text(character, persona)
    budget = _budget(context_size, reply_tokens, system, instruction or "")
    kept = _fit(messages, budget)
    with_images = _limit_images(kept)
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


def _fit(messages: list[Message], budget: int) -> list[Message]:
    """The newest messages that fit, oldest-first. The last one always does."""
    kept: list[Message] = []
    spent = 0
    for message in reversed(messages):
        cost = len(message.text) + 32 + (IMAGE_TOKENS * CHARS_PER_TOKEN if message.image else 0)
        if kept and spent + cost > budget:
            break
        kept.append(message)
        spent += int(cost)
    kept.reverse()
    return kept


def _limit_images(messages: list[Message]) -> list[tuple[Message, bool]]:
    """Which of the kept messages still carry their still."""
    allowance = MAX_IMAGES
    marked = []
    for message in reversed(messages):
        keep = bool(message.image) and allowance > 0
        allowance -= 1 if keep else 0
        marked.append((message, keep))
    marked.reverse()
    return marked


def has_image(messages: list[Message]) -> bool:
    """Whether a request built from these needs the vision projector."""
    return any(message.image for message in messages)


__all__ = ["ASSISTANT", "USER", "build", "clean_reply", "continue_instruction",
           "greeting_text", "has_image", "impersonate_instruction", "prefix_instruction",
           "substitute", "system_text"]
