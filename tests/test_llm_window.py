"""A thread's context window: where it starts, when it moves, what it keeps.

The user's report (2026-11-09): 33 seconds before the first word on about every
second message of a long thread, because the old trimmer worked the window's
front out afresh for each request, in steps of a quarter of an estimated window,
and every move of the front had llama-server read the whole conversation again.
These hold the rules that replace it (``mc_llm_window``): the start is kept and
moves only when the window no longer fits; it moves far enough to leave a
quarter of the room; it never cuts into the last exchange; the room kept for a
reply follows the thread's replies; and Trim keeps whole messages.
"""

from __future__ import annotations

import pytest

import mc_llm_paths
import mc_llm_window as window_module
from prompt_master.chat.history import ASSISTANT, USER, Message


class Counter:
    """Four characters a token and eight a message, like the real overhead."""

    exact = True

    def text(self, text):
        return len(text or "") // 4

    def message(self, text, still=False):
        return self.text(text) + 8 + (300 if still else 0)


def said(role, tokens):
    """A message of ``tokens`` text tokens at the counter above."""
    return Message(role=role, versions=["x" * (4 * tokens)])


def thread(*sizes):
    """Alternating your messages and replies, of these many text tokens each."""
    return [said(USER if index % 2 == 0 else ASSISTANT, size)
            for index, size in enumerate(sizes)]


@pytest.fixture(autouse=True)
def root(tmp_path, monkeypatch):
    monkeypatch.setattr(mc_llm_paths, "data_root", lambda: tmp_path)
    return tmp_path


class TestTheKeptWindow:
    def test_a_thread_with_none_starts_from_its_first_message(self):
        assert window_module.for_thread("Ada", "t1") == window_module.Window()

    def test_a_start_is_kept_beside_the_threads_not_in_them(self, root):
        window_module.set_front("Ada", "t1", 6)

        assert window_module.for_thread("Ada", "t1").front == 6
        assert (root / window_module.FILE_NAME).is_file()
        assert window_module.for_thread("Ada", "t2").front == 0

    def test_a_summary_starts_the_window_where_it_ends(self):
        window_module.set_summary("Ada", "t1", "They met.", upto=8, tokens=3)

        kept = window_module.for_thread("Ada", "t1")
        assert (kept.summary, kept.upto, kept.summary_tokens, kept.front) == (
            "They met.", 8, 3, 8)
        assert window_module.summary_block(kept).startswith(window_module.SUMMARY_HEAD)

    def test_clear_forgets_the_start_and_the_summary_and_keeps_the_measured_ratio(self):
        window_module.set_summary("Ada", "t1", "They met.", upto=8, tokens=3)
        window_module.remember_ratio("Ada", "t1", 0.25)

        window_module.clear("Ada", "t1")

        kept = window_module.for_thread("Ada", "t1")
        assert (kept.front, kept.summary, kept.upto) == (0, "", 0)
        assert kept.ratio == pytest.approx(0.25)

    def test_a_deleted_thread_forgets_its_window(self):
        window_module.set_front("Ada", "t1", 4)

        window_module.forget("Ada", "t1")

        assert window_module.for_thread("Ada", "t1") == window_module.Window()

    def test_a_ratio_that_barely_moved_is_not_written_again(self, root):
        window_module.remember_ratio("Ada", "t1", 0.25)
        before = (root / window_module.FILE_NAME).stat().st_mtime_ns

        window_module.remember_ratio("Ada", "t1", 0.251)

        assert (root / window_module.FILE_NAME).stat().st_mtime_ns == before


class TestDeletingMessages:
    def test_a_message_deleted_before_the_start_keeps_the_window_on_the_same_messages(self):
        window_module.set_front("Ada", "t1", 6)

        window_module.removed("Ada", "t1", index=2, count=1, remaining=9)

        assert window_module.for_thread("Ada", "t1").front == 5

    def test_a_message_deleted_after_the_start_moves_nothing(self):
        window_module.set_front("Ada", "t1", 6)

        window_module.removed("Ada", "t1", index=7, count=1, remaining=9)

        assert window_module.for_thread("Ada", "t1").front == 6

    def test_one_deleted_inside_the_summary_shifts_what_it_covers(self):
        window_module.set_summary("Ada", "t1", "They met.", upto=8, tokens=3)

        window_module.removed("Ada", "t1", index=3, count=1, remaining=11)

        kept = window_module.for_thread("Ada", "t1")
        assert (kept.summary, kept.upto, kept.front) == ("They met.", 7, 7)

    def test_delete_from_here_into_the_summary_drops_it(self):
        """A summary of a conversation that, after the rewind, did not happen."""
        window_module.set_summary("Ada", "t1", "They met.", upto=8, tokens=3)

        window_module.removed("Ada", "t1", index=5, count=7, remaining=5)

        kept = window_module.for_thread("Ada", "t1")
        assert (kept.summary, kept.upto, kept.front) == ("", 0, 0)


class TestABranch:
    def test_a_branch_after_the_start_keeps_the_window_and_the_summary(self):
        window_module.set_summary("Ada", "src", "They met.", upto=4, tokens=3)
        window_module.set_front("Ada", "src", 6)

        window_module.carry("Ada", "src", "made", 9)

        kept = window_module.for_thread("Ada", "made")
        assert (kept.summary, kept.upto, kept.front) == ("They met.", 4, 6)

    def test_a_branch_before_the_start_starts_after_the_summary(self):
        window_module.set_summary("Ada", "src", "They met.", upto=4, tokens=3)
        window_module.set_front("Ada", "src", 8)

        window_module.carry("Ada", "src", "made", 5)

        kept = window_module.for_thread("Ada", "made")
        assert (kept.summary, kept.front) == ("They met.", 4)

    def test_a_branch_inside_the_summary_does_not_take_it(self):
        window_module.set_summary("Ada", "src", "They met.", upto=8, tokens=3)

        window_module.carry("Ada", "src", "made", 3)

        kept = window_module.for_thread("Ada", "made")
        assert (kept.summary, kept.front) == ("", 0)


class TestTheRoomKeptForAReply:
    def test_a_thread_with_no_reply_keeps_oobaboogas_default(self):
        assert window_module.reserve(thread(40), 2048, Counter()) == window_module.RESERVE_NEW

    def test_it_follows_the_longest_recent_reply(self):
        history = thread(20, 300, 20, 500, 20, 420)

        assert window_module.reserve(history, 2048, Counter()) == 625

    def test_never_less_than_the_floor(self):
        history = thread(20, 50, 20, 60)

        assert window_module.reserve(history, 2048, Counter()) == window_module.RESERVE_FLOOR

    def test_never_more_than_the_threads_reply_tokens(self):
        history = thread(20, 900)

        assert window_module.reserve(history, 512, Counter()) == 512

    def test_only_the_last_few_replies_count(self):
        history = thread(20, 1500, *([20, 100] * window_module.RECENT_REPLIES))

        assert window_module.reserve(history, 4096, Counter()) == window_module.RESERVE_FLOOR


class TestWhereARequestsWindowStarts:
    def plan(self, history, window=None, context=2000, limit=512, fixed=50, ahead=0):
        return window_module.plan(history, window or window_module.Window(), Counter(), fixed,
                                  context, limit, ahead=ahead)

    def test_a_conversation_that_fits_is_sent_whole(self):
        found = self.plan(thread(100, 200, 100))

        assert (found.front, found.moved) == (0, False)
        assert found.total == 50 + 108 + 208 + 108

    def test_a_kept_start_is_where_it_starts_while_it_fits(self):
        """Even where more would fit: a window that grew back would change the
        prompt's beginning, and the cache with it."""
        history = thread(100, 200, 100, 200, 100)

        found = self.plan(history, window_module.Window(front=2))

        assert (found.front, found.moved) == (2, False)

    def test_past_the_room_it_moves_to_three_quarters_of_it(self):
        history = thread(*([100, 300] * 6), 100)   # 13 messages, about 2,700 tokens

        found = self.plan(history, context=2000)

        room = 2000 - found.reserve
        assert found.moved
        assert found.total <= room * (1 - window_module.STEP)
        # And no further than that: one message less and it would be over.
        assert found.total + found.costs[found.front - 1] > room * (1 - window_module.STEP)

    def test_it_never_moves_past_your_last_message(self):
        history = thread(100, 300, 2000, 1500)   # the last exchange alone is over the room

        found = self.plan(history, context=1000)

        assert found.front == 2
        assert history[found.front].role == USER

    def test_room_for_your_next_message_can_move_it_now(self):
        history = thread(*([100, 300] * 4))   # about 1,700 tokens, ending on a reply

        still = self.plan(history, context=2200)
        ahead = self.plan(history, context=2200, ahead=400)

        assert not still.moved
        assert ahead.moved and ahead.front > 0
        assert ahead.total + 400 <= ahead.room

    def test_a_summary_is_never_sent_twice_by_a_window_that_starts_inside_it(self):
        history = thread(100, 200, 100, 200, 100, 200, 100)
        kept = window_module.Window(front=0, summary="They met.", upto=4)

        assert window_module.start_of(kept, history) == 4


class TestKeepTheLatest:
    def test_whole_messages_that_fit(self):
        history = thread(100, 300, 100, 300, 100, 300)

        front, kept = window_module.keep_latest(history, window_module.Window(), Counter(), 900)

        assert front == 2
        assert kept == 108 + 308 + 108 + 308
        assert kept <= 900 < kept + 308   # the message before would not have fitted

    def test_the_last_exchange_whatever_it_costs(self):
        history = thread(100, 300, 100, 2000)   # the last reply alone is over what is kept

        front, kept = window_module.keep_latest(history, window_module.Window(), Counter(), 50)

        assert front == 2
        assert kept == 108 + 2008

    def test_further_back_than_the_window_starts_now_when_asked(self):
        history = thread(100, 300, 100, 300, 100, 300)

        front, _ = window_module.keep_latest(history, window_module.Window(front=4), Counter(),
                                             10_000)

        assert front == 0

    def test_never_back_into_a_summary(self):
        history = thread(100, 300, 100, 300, 100, 300)
        kept = window_module.Window(front=2, summary="They met.", upto=2)

        front, _ = window_module.keep_latest(history, kept, Counter(), 10_000)

        assert front == 2


class TestLookingAheadAfterAReply:
    def test_a_window_that_fits_but_would_not_with_your_next_message_moves_now(self):
        history = thread(*([100, 300] * 4))   # ends on a reply, fits a 2,200 context
        kept = window_module.Window()

        now = window_module.plan(history, kept, Counter(), 50, 2200, 512)
        ahead = window_module.lookahead(history, kept, Counter(), 50, 2200, 512)

        assert not now.moved
        assert ahead.moved and ahead.front > 0
        assert ahead.total + window_module.allowance(history, Counter()) <= ahead.room


class TestTheAllowanceForYourNextMessage:
    def test_at_least_the_floor(self):
        assert window_module.allowance(thread(10, 300), Counter()) == \
            window_module.ALLOWANCE_FLOOR

    def test_twice_your_longest_recent_message(self):
        assert window_module.allowance(thread(150, 300, 400, 300), Counter()) == 800
