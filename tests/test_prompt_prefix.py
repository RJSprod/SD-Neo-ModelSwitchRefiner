"""The front of the prompt stays put from one turn to the next.

llama-server resumes a prompt at the longest prefix it already has in its
cache and reads only what follows, so a turn costs what is new at the end --
unless something near the start changed, in which case everything after that
point is read again. From the log that prompted this: a prompt of five
thousand tokens read from the first token on every turn (``checking sim =
335/4942``, then ``92/5241``), on a GPU reading 25 to 50 tokens a second, was
three minutes of "Replying…" before a word appeared. Two things in the builder
moved the front every turn once a conversation was long enough: the window
dropped exactly one message from the front per turn, and every new picture
took the still off the *oldest* picture message, rewriting it.

Every test here was checked against the change it guards by reverting that
change and watching the test fail.
"""

from __future__ import annotations

from prompt_master.chat import prompt as builder
from prompt_master.chat.characters import Character, Persona
from prompt_master.chat.history import ASSISTANT, USER, Message
from prompt_master.chat.prompt import (IMAGE_NOTE, MAX_IMAGES, TRIM_STEP, _fit, build,
                                       stills_carried)

C, P = Character(name="C"), Persona(name="P")


def turn(index: int, image: str = "") -> Message:
    who = USER if index % 2 == 0 else ASSISTANT
    return Message(role=who, versions=[f"turn {index} " + "x" * 200], image=image,
                   image_name=f"picture-{index}.png" if image else "")


def still(index: int) -> str:
    return "data:image/png;base64,%04d" % index


def conversation(turns: int) -> list[Message]:
    return [turn(index) for index in range(turns)]


def cost(messages) -> int:
    return sum(builder._cost(message) for message in messages)


class TestTheWindowIsTrimmedInSteps:
    def test_a_history_that_fits_is_sent_whole(self):
        messages = conversation(10)

        assert _fit(messages, 10 ** 6) == messages

    def test_a_trim_cuts_at_a_boundary_rather_than_at_the_budget(self):
        messages = conversation(40)
        budget = cost(messages[-20:]) + 100      # the newest twenty fit, and no more

        kept = _fit(messages, budget)

        assert kept == messages[-len(kept):], "the newest, oldest-first"
        assert cost(kept) <= budget
        assert len(kept) < 20, "cut at the next boundary past the newest that fit, not at them"

    def test_the_front_holds_still_and_then_moves_by_a_step(self):
        """Once the history no longer fits, the first message sent is the
        same one turn after turn until the newest that fit have walked a step
        past it, and then it moves by several messages at once. The old rule
        moved it by one message every turn: twenty fronts in twenty turns."""
        history = conversation(40)
        budget = cost(history[-20:]) + 100
        fronts = []
        for index in range(40, 60):
            fronts.append(_fit(history, budget)[0].text)
            history.append(turn(index))

        runs = []
        for front in fronts:
            if runs and runs[-1][0] == front:
                runs[-1][1] += 1
            else:
                runs.append([front, 1])
        assert len(runs) <= 6, f"the front moved {len(runs)} times in twenty turns"
        assert len(runs) >= 2, "a window that never moves is a window that is not trimming"
        assert max(length for _, length in runs) >= 4, runs
        step = int(budget * TRIM_STEP)
        assert step >= cost(conversation(4)), "a step is several messages"

    def test_the_last_message_is_always_sent(self):
        messages = [turn(0), turn(1)]

        assert _fit(messages, 10) == [messages[-1]]

    def test_only_the_newest_picture_is_charged_as_a_still(self):
        """An older picture costs its note, not three hundred tokens: a
        conversation with a dozen pictures in it would otherwise lose most of
        its window to stills that are not even sent."""
        noted = builder._cost(turn(0, still(0)))
        carried = builder._cost(turn(0, still(0)), still=True)
        plain = builder._cost(turn(0))

        assert carried - plain == int(builder.IMAGE_TOKENS * builder.CHARS_PER_TOKEN)
        assert noted - plain == len(IMAGE_NOTE.format(name="picture-0.png")) + 1
        history = [turn(index, still(index)) for index in range(12)]
        budget = cost(history[:-1]) + carried + 1     # room for eleven notes and one still
        assert _fit(history, budget) == history, "twelve pictures, one still, all fit"
        assert len(_fit(history, budget - 2)) < 12, "which is the whole of the budget"


class TestOnlyTheNewestPictureIsAStill:
    def test_the_stills_carried_are_one_at_most(self):
        assert MAX_IMAGES == 1
        assert stills_carried(0) == 0
        assert stills_carried(1) == 1
        assert stills_carried(5) == 1
        assert stills_carried(-3) == 0

    def test_older_pictures_become_notes_that_keep_their_text(self):
        messages = [turn(0, still(0)), turn(1), turn(2, still(2)), turn(3)]

        wire = build(C, P, messages, context_size=100000)

        older = wire[1]["content"]
        assert isinstance(older, str), "no still on the older picture's message"
        assert older.startswith(IMAGE_NOTE.format(name="picture-0.png"))
        assert "turn 0" in older
        newest = wire[3]["content"]
        assert isinstance(newest, list) and newest[0]["type"] == "image_url"
        assert newest[0]["image_url"]["url"] == still(2)
        assert builder.needs_vision(wire)

    def test_the_picture_is_kept_in_the_message_for_the_chat_to_draw(self):
        """The prompt carries the newest still and names the rest; the
        messages themselves are not touched, so the chat still draws every
        picture."""
        messages = [turn(0, still(0)), turn(1), turn(2, still(2))]

        build(C, P, messages, context_size=100000)

        assert [message.image for message in messages] == [still(0), "", still(2)]

    def test_a_new_picture_rewrites_only_the_previous_pictures_message(self):
        """Everything before the previous picture is exactly what it was, so
        the cache resumes from there rather than from the first token."""
        messages = [turn(0, still(0)), turn(1), turn(2), turn(3, still(3)), turn(4)]
        before = build(C, P, messages, context_size=100000)

        after = build(C, P, messages + [turn(5, still(5))], context_size=100000)

        assert after[:4] == before[:4]
        assert before[4]["content"][0]["type"] == "image_url"
        assert isinstance(after[4]["content"], str)
        assert after[4]["content"].startswith(IMAGE_NOTE.format(name="picture-3.png"))
        assert after[6]["content"][0]["image_url"]["url"] == still(5)


class TestThePrefixSurvivesALongConversation:
    """A hundred and fifty turns with a picture every fifth one, in an
    eight-thousand-token window that holds about sixty of them: how often is
    the whole prompt read again?"""

    @staticmethod
    def _rereads():
        history = []
        previous = None
        whole, fractions = 0, []
        for index in range(150):
            history.append(turn(index, still(index) if index % 5 == 0 else ""))
            wire = build(C, P, history, context_size=8192, reply_tokens=200)
            if previous is not None:
                shared = 0
                for mine, theirs in zip(wire, previous):
                    if mine != theirs:
                        break
                    shared += 1
                reread = sum(len(str(message["content"])) for message in wire[shared:])
                fractions.append(reread / sum(len(str(message["content"])) for message in wire))
                if shared <= 1:          # nothing past the system message survived
                    whole += 1
            previous = wire
        return whole, sum(fractions) / len(fractions)

    def test_the_whole_prompt_is_rarely_read_again(self):
        whole, average = self._rereads()

        assert whole <= 12, f"the whole prompt was read again on {whole} of 149 turns"
        assert whole >= 2, "a window that never trims is not a window"
        assert average < 0.25, f"on average {average:.0%} of the prompt was read again each turn"
