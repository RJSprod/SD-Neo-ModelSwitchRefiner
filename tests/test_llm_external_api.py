"""The external MiniMax H3 API: the contract another extension is written against.

Every test here is a sentence from ``docs/21-external-llm-api.md``, and that is
deliberate. The other end of this API is code somebody else maintains in a
different repository; the documentation is the only thing they have, so a
promise made there and not checked here is a promise that will be broken by
accident.

The queue is stepped by hand throughout. ``mc_llm_jobs`` runs a background
worker in the host, but a test that asserted "this job is at position 2" while
a thread was draining it would fail once a fortnight for reasons nobody could
reproduce -- so ``conftest``'s autouse fixture pauses the worker and these tests
call :func:`mc_llm_jobs.drain_once` at the moments they mean.

What is *not* re-tested here is the enhancement itself. ``mc_llm_sessions
.minimax`` is replaced with a generator of known events, because what this
module adds is a queue, a record and a feed around that generator -- and a test
that also needed a llama-server would be testing the wrong half.
"""

from __future__ import annotations

import time

import pytest

import mc_llm_api as api
import mc_llm_jobs as jobs
import mc_llm_minimax_panel
import mc_llm_sessions as sessions
import mc_llm_state
from prompt_master.minimax import enhancer


# --------------------------------------------------------------------------- #
# Doubles
# --------------------------------------------------------------------------- #


def enhancement(text: str = "Shot one.", *, caption: str = "", chunks: int = 2,
                fail: str = "", record: list | None = None):
    """A stand-in for one run of the enhancer, in the real event order.

    The order is the product -- status, then caption if there was a picture,
    then chunks, then a terminal event -- so a double that emitted them in any
    other order would let a consumer that depends on the real one pass.
    """

    def double(prompt, variant, image, seed, cancel, system=None):
        if record is not None:
            record.append({"prompt": prompt, "variant": variant, "image": image,
                           "seed": seed, "system": system, "cancel": cancel})
        yield sessions.Event(sessions.STATUS, "Preparing the model…")
        if image is not None and caption:
            yield sessions.Event(sessions.CAPTION, caption)
        for index in range(chunks):
            if cancel.is_set():
                yield sessions.Event(sessions.CANCELLED, "Cancelled")
                return
            yield sessions.Event(sessions.CHUNK, f"part{index} ")
        if fail:
            yield sessions.Event(sessions.FAILED, fail)
            return
        if cancel.is_set():
            yield sessions.Event(sessions.CANCELLED, "Cancelled")
            return
        yield sessions.Event(sessions.DONE, text)

    return double


def blocking_enhancement(started, release, record: list | None = None):
    """A run that parks until a test lets it go. For "in progress, not interrupted"."""

    def double(prompt, variant, image, seed, cancel, system=None):
        if record is not None:
            record.append(prompt)
        yield sessions.Event(sessions.STATUS, "Working…")
        started.set()
        while not release.wait(timeout=0.02):
            if cancel.is_set():
                yield sessions.Event(sessions.CANCELLED, "Cancelled")
                return
        yield sessions.Event(sessions.DONE, f"done: {prompt}")

    return double


class FakeClient:
    """A llama.cpp client that answers instantly and remembers the asking.

    The same double ``test_krea_creative`` uses, restated rather than imported:
    what these tests read off it is the *system message*, which is the one part
    of a request the external API can change.
    """

    def __init__(self, answers=None):
        self.answers = list(answers or [])
        self.calls: list[dict] = []

    def stream_chat(self, messages, max_tokens, seed, on_text, cancel=None,
                    temperature=0.6, top_p=0.9):
        self.calls.append({"messages": messages, "seed": seed})
        answer = self.answers.pop(0) if self.answers else "An H3 prompt."
        on_text(answer)
        return answer

    @property
    def system(self) -> str:
        return self.calls[-1]["messages"][0]["content"]


@pytest.fixture
def client(monkeypatch, host):
    """The real :func:`mc_llm_sessions.minimax`, over a client that answers instantly.

    Every other test in this file replaces ``sessions.minimax`` wholesale, which
    is right for testing a queue and wrong for testing what the queue *sends*.
    These few go the whole way down, because the override is only real if it
    reaches the messages.
    """
    import mc_broker

    mc_broker.clear()
    monkeypatch.setattr(mc_broker, "host_busy", lambda: False)
    fake = FakeClient()
    monkeypatch.setattr(sessions, "_client",
                        lambda *args, **kwargs: fake)
    monkeypatch.setattr(sessions, "_placement_notes", lambda role="": [])
    monkeypatch.setattr(sessions, "_preparing", lambda role="": "Preparing")
    yield fake
    mc_broker.clear()


@pytest.fixture(autouse=True)
def store(tmp_path, monkeypatch, host):
    """A throwaway data directory, so the history these jobs file is this test's."""
    import mc_llm_paths

    monkeypatch.setattr(mc_llm_paths, "data_root", lambda: tmp_path)
    return tmp_path


@pytest.fixture
def written(monkeypatch):
    """The default double: every job finishes with the same short prompt."""
    monkeypatch.setattr(sessions, "minimax", enhancement())


def picture(tmp_path, name="frame.png", colour=(200, 30, 30)):
    from PIL import Image

    path = tmp_path / name
    Image.new("RGB", (24, 24), colour).save(path)
    return path


# --------------------------------------------------------------------------- #
# Submitting
# --------------------------------------------------------------------------- #


class TestAsking:
    """T1 — what a caller may ask for, and what it is told when it may not."""

    def test_a_prompt_is_required(self):
        with pytest.raises(api.Rejected) as raised:
            api.submit_minimax("   ")

        assert raised.value.code == "empty_prompt"

    def test_a_request_comes_back_with_an_id_and_a_place_in_the_line(self, written):
        identifier = api.submit_minimax("a car chase at dusk")

        assert jobs.position_of(identifier) == 1
        assert api.status(identifier)["state"] == jobs.QUEUED

    def test_an_unknown_variant_resolves_to_the_one_wangp_lists_first(self, written):
        identifier = api.submit_minimax("a car chase", variant="h4")

        assert api.status(identifier)["variant"] == enhancer.FL2VA

    def test_the_variant_is_read_case_insensitively(self, written):
        identifier = api.submit_minimax("a car chase", variant="REF2VA")

        assert api.status(identifier)["variant"] == enhancer.REF2VA

    def test_a_seed_is_drawn_per_request_unless_one_is_given(self, written):
        drawn = api.status(api.submit_minimax("one"))["seed"]
        given = api.status(api.submit_minimax("two", seed=4242))["seed"]

        assert given == 4242
        assert drawn != 0

    def test_the_origin_is_recorded_so_the_panel_can_name_who_is_asking(self, written):
        identifier = api.submit_minimax("one", origin="video-tools")

        assert api.status(identifier)["origin"] == "video-tools"

    def test_a_full_queue_is_refused_rather_than_grown(self, written, monkeypatch):
        monkeypatch.setattr(jobs, "MAX_QUEUED", 3)
        for index in range(3):
            api.submit_minimax(f"prompt {index}")

        with pytest.raises(api.Rejected) as raised:
            api.submit_minimax("one too many")

        assert raised.value.code == "queue_full"
        assert jobs.snapshot()["waiting"] == 3

    def test_status_for_a_forgotten_request_is_none_rather_than_an_error(self):
        assert api.status("never-existed") is None


# --------------------------------------------------------------------------- #
# Pictures
# --------------------------------------------------------------------------- #


class TestPictures:
    """T2 — three slots, one caption, and a record of which was which.

    The vendored path captions exactly one image and hands the enhancer one
    ``image_caption:`` line. A caller may fill any combination of the three
    slots; what it must never have to do is guess which of its pictures the
    prompt was written about.
    """

    def test_fl2va_captions_the_first_frame_when_it_has_one(self, written, tmp_path):
        identifier = api.submit_minimax(
            "a car chase", variant=enhancer.FL2VA,
            first_frame=picture(tmp_path, "first.png"),
            last_frame=picture(tmp_path, "last.png", (30, 200, 30)),
            reference=picture(tmp_path, "ref.png", (30, 30, 200)))

        found = api.status(identifier)
        assert found["image_used"] == api.FIRST_FRAME
        assert sorted(found["image_ignored"]) == [api.LAST_FRAME, api.REFERENCE]

    def test_ref2va_captions_the_reference_instead(self, written, tmp_path):
        identifier = api.submit_minimax(
            "a car chase", variant=enhancer.REF2VA,
            first_frame=picture(tmp_path, "first.png"),
            reference=picture(tmp_path, "ref.png", (30, 30, 200)))

        assert api.status(identifier)["image_used"] == api.REFERENCE

    def test_a_variant_falls_back_to_a_slot_it_would_not_have_chosen(self, written,
                                                                     tmp_path):
        """A caller that filled only the "wrong" slot still gets a picture used.

        The alternative -- writing a text-only prompt because FL2VA was asked
        for and only a reference was sent -- would silently ignore the one thing
        the caller bothered to attach.
        """
        identifier = api.submit_minimax(
            "a car chase", variant=enhancer.FL2VA,
            reference=picture(tmp_path, "ref.png"))

        assert api.status(identifier)["image_used"] == api.REFERENCE

    def test_no_picture_means_no_caption_and_an_empty_record(self, written):
        found = api.status(api.submit_minimax("a car chase"))

        assert found["image_used"] == ""
        assert found["images"] == []

    def test_exactly_one_picture_reaches_the_enhancer(self, monkeypatch, tmp_path):
        seen: list = []
        monkeypatch.setattr(sessions, "minimax", enhancement(record=seen))
        api.submit_minimax("a car chase", variant=enhancer.FL2VA,
                           first_frame=picture(tmp_path, "first.png"),
                           last_frame=picture(tmp_path, "last.png", (30, 200, 30)))
        jobs.drain_once()

        assert seen[0]["image"].startswith("data:image/")

    @pytest.mark.parametrize("shape", ["path", "string", "bytes", "pil", "data_url"])
    def test_every_shape_a_caller_might_hold_a_picture_in_is_accepted(self, written,
                                                                      tmp_path, shape):
        """Four shapes and a pass-through, because a caller in this process has all of them.

        The point of accepting a decoded ``PIL.Image`` is the one the panel
        already makes: writing it to a temporary file so it could be read back
        would be two extra copies of somebody's photograph on their disk.
        """
        from PIL import Image

        path = picture(tmp_path)
        supplied = {
            "path": path,
            "string": str(path),
            "bytes": path.read_bytes(),
            "pil": Image.open(path).copy(),
            "data_url": api._data_url(path, api.FIRST_FRAME),
        }[shape]

        identifier = api.submit_minimax("a car chase", first_frame=supplied)

        assert api.status(identifier)["image_used"] == api.FIRST_FRAME

    def test_an_unreadable_picture_is_refused_at_the_call_site(self, written, tmp_path):
        broken = tmp_path / "broken.png"
        broken.write_bytes(b"not a picture")

        with pytest.raises(api.Rejected) as raised:
            api.submit_minimax("a car chase", first_frame=broken)

        assert raised.value.code == "bad_image"
        assert "first frame" in str(raised.value)

    def test_no_picture_bytes_are_kept_in_the_record(self, written, tmp_path):
        """A record somebody may log must not contain a base64 photograph.

        The same rule the Krea history follows. The data URL exists for exactly
        as long as the run needs it.
        """
        identifier = api.submit_minimax("a car chase",
                                        first_frame=picture(tmp_path))
        jobs.drain_once()

        assert "data:image" not in repr(api.status(identifier))
        assert jobs.job(identifier)._image is None


# --------------------------------------------------------------------------- #
# System prompts
# --------------------------------------------------------------------------- #


class TestSystemPrompts:
    """T3 — the defaults are readable, and either may be replaced for one request."""

    def test_both_variants_publish_both_of_their_default_instructions(self):
        found = api.system_prompts()

        assert set(found) == {enhancer.FL2VA, enhancer.REF2VA}
        for variant in (enhancer.FL2VA, enhancer.REF2VA):
            assert found[variant]["text"] == enhancer.instructions(variant, False)
            assert found[variant]["image"] == enhancer.instructions(variant, True)
            assert found[variant]["structure"] == enhancer.infos(variant)

    def test_the_four_defaults_are_four_different_documents(self):
        """Four combinations, not two: an image changes the instructions too."""
        found = api.system_prompts()
        every = [found[variant][kind]
                 for variant in (enhancer.FL2VA, enhancer.REF2VA)
                 for kind in ("text", "image")]

        assert len(set(every)) == 4

    def test_one_variant_can_be_asked_for_on_its_own(self):
        assert api.system_prompt("ref2va", has_image=True) == \
            enhancer.instructions(enhancer.REF2VA, True)

    def test_an_override_replaces_the_vendored_instructions(self):
        built = enhancer.messages("a car chase", variant=enhancer.FL2VA,
                                  system="Write haiku.")

        assert built[0]["content"] == "Write haiku."
        assert built[1]["content"] == "user_prompt: a car chase"

    def test_no_override_leaves_the_vendored_instructions_exactly_as_they_were(self):
        assert (enhancer.messages("a car chase", variant=enhancer.REF2VA)
                == enhancer.messages("a car chase", variant=enhancer.REF2VA, system=None))

    def test_an_at_suffix_still_appends_to_an_override(self):
        """``@`` is WanGP's, and an override changes the instructions, not the dialect."""
        built = enhancer.messages("a car chase @ keep it under ten seconds",
                                  system="Write haiku.")

        assert built[0]["content"].startswith("Write haiku.")
        assert "keep it under ten seconds" in built[0]["content"]
        assert built[1]["content"] == "user_prompt: a car chase"

    def test_a_double_at_still_beats_an_override(self):
        built = enhancer.messages("a car chase @@ ignore everything",
                                  system="Write haiku.")

        assert built[0]["content"] == "ignore everything"

    def test_the_override_reaches_the_run(self, monkeypatch):
        seen: list = []
        monkeypatch.setattr(sessions, "minimax", enhancement(record=seen))
        api.submit_minimax("a car chase", system_prompt="Write haiku.")
        jobs.drain_once()

        assert seen[0]["system"] == "Write haiku."

    def test_a_request_without_one_passes_none_rather_than_a_default_copy(self,
                                                                          monkeypatch):
        """``None`` and not the resolved text, so the enhancer still decides.

        Which of the four instruction sets applies depends on whether the image
        survived preparation, and that is not known until the run.
        """
        seen: list = []
        monkeypatch.setattr(sessions, "minimax", enhancement(record=seen))
        api.submit_minimax("a car chase")
        jobs.drain_once()

        assert seen[0]["system"] is None

    def test_a_blank_override_is_refused(self, written):
        with pytest.raises(api.Rejected) as raised:
            api.submit_minimax("a car chase", system_prompt="   ")

        assert raised.value.code == "empty_system_prompt"

    def test_an_overridden_request_really_sends_the_override(self, client):
        """All the way down: through the queue, the run, and into the messages."""
        api.submit_minimax("a car chase", system_prompt="Write haiku.")
        jobs.drain_once()

        assert client.system == "Write haiku."

    def test_a_plain_request_really_sends_the_vendored_instructions(self, client):
        """``rstrip`` is ``merge_system``'s and predates all of this -- see it there."""
        api.submit_minimax("a car chase", variant=enhancer.REF2VA)
        jobs.drain_once()

        assert client.system == enhancer.instructions(enhancer.REF2VA, False).rstrip()

    def test_the_user_turn_is_wangps_labelled_line_either_way(self, client):
        api.submit_minimax("a car chase", system_prompt="Write haiku.")
        jobs.drain_once()

        assert client.calls[-1]["messages"][-1]["content"] == "user_prompt: a car chase"

    def test_the_record_says_whether_the_instructions_were_the_defaults(self, written):
        plain = api.submit_minimax("one")
        overridden = api.submit_minimax("two", system_prompt="Write haiku.")

        assert api.status(plain)["system_override"] is False
        assert api.status(overridden)["system_override"] is True


# --------------------------------------------------------------------------- #
# The queue
# --------------------------------------------------------------------------- #


class TestTheQueue:
    """T4 — arriving does not preempt, and the line is first in, first out."""

    def test_requests_run_in_the_order_they_arrived(self, monkeypatch):
        seen: list = []
        monkeypatch.setattr(sessions, "minimax", enhancement(record=seen))
        for index in range(3):
            api.submit_minimax(f"prompt {index}")
        while jobs.drain_once():
            pass

        assert [found["prompt"] for found in seen] == ["prompt 0", "prompt 1", "prompt 2"]

    def test_a_request_arriving_mid_run_waits_rather_than_interrupting(self, monkeypatch):
        """The sentence the whole feature was asked for. A run in progress is not touched."""
        import threading

        started, release = threading.Event(), threading.Event()
        monkeypatch.setattr(sessions, "minimax", blocking_enhancement(started, release))
        first = api.submit_minimax("the one already running")
        worker = threading.Thread(target=jobs.drain_once, daemon=True)
        worker.start()
        assert started.wait(timeout=5)

        second = api.submit_minimax("the one that arrived during it")

        assert api.status(first)["state"] == jobs.RUNNING
        assert api.status(second)["state"] == jobs.QUEUED
        assert jobs.position_of(second) == 1
        release.set()
        worker.join(timeout=5)
        assert api.status(first)["state"] == jobs.DONE

    def test_positions_move_up_as_the_line_shortens(self, written):
        first = api.submit_minimax("one")
        second = api.submit_minimax("two")
        third = api.submit_minimax("three")

        assert (jobs.position_of(second), jobs.position_of(third)) == (2, 3)
        jobs.drain_once()
        assert jobs.position_of(first) == 0
        assert (jobs.position_of(second), jobs.position_of(third)) == (1, 2)

    def test_a_waiting_request_is_told_when_its_position_changes(self, written):
        """Emitted rather than left to be polled: it is the number that changes
        while the caller is not looking."""
        api.submit_minimax("one")
        second = api.submit_minimax("two")
        feed = api.subscribe(second)
        jobs.drain_once()

        moved = [event for event in feed.poll() if event["event"] == jobs.EV_POSITION]

        assert moved and moved[-1]["data"]["position"] == 1

    def test_a_position_is_announced_once_per_move_and_not_once_per_event(self, written):
        """A subscriber that already knows it is second must not be told twice.

        The line shifts when a job is taken off it, and it does not shift again
        when that job finishes -- but both moments call the announcer, because
        both *can* shift it.
        """
        api.submit_minimax("one")
        second = api.submit_minimax("two")
        feed = api.subscribe(second)
        jobs.drain_once()

        moved = [event for event in feed.poll() if event["event"] == jobs.EV_POSITION]

        assert [event["data"]["position"] for event in moved] == [1]

    def test_cancelling_out_of_the_middle_moves_everyone_behind_it_up(self, written):
        api.submit_minimax("one")
        second = api.submit_minimax("two")
        third = api.submit_minimax("three")
        feed = api.subscribe(third)
        feed.poll()

        api.cancel(second)

        moved = [event for event in feed.poll() if event["event"] == jobs.EV_POSITION]
        assert [event["data"]["position"] for event in moved] == [2]
        assert jobs.position_of(third) == 2

    def test_the_snapshot_is_one_consistent_read(self, written):
        api.submit_minimax("one", origin="video-tools")
        api.submit_minimax("two")

        found = api.queue()

        assert found["active"] is True
        assert found["waiting"] == 2
        assert found["running"] is None
        assert [entry["origin"] for entry in found["queue"]] == ["video-tools", ""]

    def test_the_queue_listing_does_not_carry_everyone_elses_prompts(self, written):
        """A caller polling for position should not be handed the answers."""
        api.submit_minimax("one")
        jobs.drain_once()

        assert "prompt" not in api.queue()["recent"][0]

    def test_a_finished_request_keeps_its_answer_until_it_is_forgotten(self, written):
        identifier = api.submit_minimax("one")
        jobs.drain_once()

        assert api.result(identifier) == "Shot one."
        assert api.forget(identifier) is True
        assert api.status(identifier) is None

    def test_an_unfinished_request_is_not_forgotten_on_request(self, written):
        identifier = api.submit_minimax("one")

        assert api.forget(identifier) is False

    def test_a_failure_is_recorded_as_one_and_does_not_stop_the_queue(self, monkeypatch):
        monkeypatch.setattr(sessions, "minimax", enhancement(fail="no model is loaded"))
        first = api.submit_minimax("one")
        second = api.submit_minimax("two")
        while jobs.drain_once():
            pass

        assert api.status(first)["state"] == jobs.FAILED
        assert api.status(first)["error"] == "no model is loaded"
        assert api.status(second)["state"] == jobs.FAILED

    def test_a_run_that_raises_becomes_a_failed_job_rather_than_a_stuck_one(self,
                                                                            monkeypatch):
        def explode(*args, **kwargs):
            raise RuntimeError("the server went away")
            yield  # pragma: no cover -- makes this a generator

        monkeypatch.setattr(sessions, "minimax", explode)
        identifier = api.submit_minimax("one")
        jobs.drain_once()

        assert api.status(identifier)["state"] == jobs.FAILED
        assert "the server went away" in api.status(identifier)["error"]
        assert jobs.active() is False


# --------------------------------------------------------------------------- #
# Cancelling
# --------------------------------------------------------------------------- #


class TestCancelling:
    """T5 — a caller may stop its own request wherever it is."""

    def test_a_queued_request_leaves_the_line_without_touching_the_card(self, monkeypatch):
        seen: list = []
        monkeypatch.setattr(sessions, "minimax", enhancement(record=seen))
        first = api.submit_minimax("one")
        second = api.submit_minimax("two")

        found = api.cancel(second, "changed my mind")

        assert found == {"ok": True, "state": jobs.CANCELLED, "was": jobs.QUEUED}
        assert api.status(second)["reason"] == "changed my mind"
        while jobs.drain_once():
            pass
        assert [entry["prompt"] for entry in seen] == ["one"]
        assert api.status(first)["state"] == jobs.DONE

    def test_a_running_request_is_asked_to_stop_and_says_cancelling(self, monkeypatch):
        import threading

        started, release = threading.Event(), threading.Event()
        monkeypatch.setattr(sessions, "minimax", blocking_enhancement(started, release))
        identifier = api.submit_minimax("one")
        worker = threading.Thread(target=jobs.drain_once, daemon=True)
        worker.start()
        assert started.wait(timeout=5)

        found = api.cancel(identifier, "stopped")

        assert found == {"ok": True, "state": "cancelling", "was": jobs.RUNNING}
        worker.join(timeout=5)
        assert api.status(identifier)["state"] == jobs.CANCELLED

    def test_a_finished_request_cannot_be_cancelled(self, written):
        identifier = api.submit_minimax("one")
        jobs.drain_once()

        found = api.cancel(identifier)

        assert found["ok"] is False
        assert found["code"] == "already_finished"

    def test_an_unknown_id_is_refused_by_name(self):
        assert api.cancel("nope")["code"] == "unknown_job"

    def test_cancel_all_empties_the_queue(self, written):
        for index in range(3):
            api.submit_minimax(f"prompt {index}")

        found = jobs.cancel_all("clearing out")

        assert found["cancelled"] == 3
        assert jobs.active() is False

    def test_a_request_cancelled_before_it_ran_never_reaches_the_enhancer(self,
                                                                          monkeypatch):
        """Not merely marked cancelled -- the double must never be called at all."""
        seen: list = []
        monkeypatch.setattr(sessions, "minimax", enhancement(record=seen))
        identifier = api.submit_minimax("one")
        api.cancel(identifier)
        jobs.drain_once()

        assert seen == []

    def test_a_finished_id_left_in_the_line_is_skipped_rather_than_run(self, monkeypatch):
        """The guard at the mouth of the worker, tested as the guard it is.

        Cancelling takes a request out of the deque, so this state is one the
        module should never reach -- which is exactly why the worker checks. A
        terminal job re-run would write a second answer over a caller's first,
        and the cost of being sure is one comparison per request.
        """
        seen: list = []
        monkeypatch.setattr(sessions, "minimax", enhancement(record=seen))
        identifier = api.submit_minimax("one")
        api.cancel(identifier)
        jobs._pending.append(identifier)

        assert jobs.drain_once() is True
        assert seen == []
        assert api.status(identifier)["state"] == jobs.CANCELLED


# --------------------------------------------------------------------------- #
# Feeds
# --------------------------------------------------------------------------- #


class TestFeeds:
    """T6 — subscribing, resuming, and the lifecycle that stops one leaking."""

    def test_a_feed_carries_the_run_in_order(self, monkeypatch):
        monkeypatch.setattr(sessions, "minimax", enhancement(caption="a red car",
                                                             chunks=2))
        identifier = api.submit_minimax("one", first_frame=None)
        feed = api.subscribe(identifier)
        jobs.drain_once()

        names = [event["event"] for event in feed.poll()]

        assert names == [jobs.EV_QUEUED, jobs.EV_STARTED, jobs.EV_STATUS,
                         jobs.EV_CHUNK, jobs.EV_CHUNK, jobs.EV_DONE]

    def test_the_caption_arrives_as_its_own_event_before_the_prompt(self, monkeypatch,
                                                                    tmp_path):
        monkeypatch.setattr(sessions, "minimax", enhancement(caption="a red car"))
        identifier = api.submit_minimax("one", first_frame=picture(tmp_path))
        feed = api.subscribe(identifier)
        jobs.drain_once()

        names = [event["event"] for event in feed.poll()]

        assert names.index(jobs.EV_CAPTION) < names.index(jobs.EV_CHUNK)
        assert api.status(identifier)["caption"] == "a red car"

    def test_a_feed_opened_late_still_replays_from_the_beginning(self, written):
        """The gap between submitting and subscribing has to be harmless."""
        identifier = api.submit_minimax("one")
        jobs.drain_once()

        feed = api.subscribe(identifier)

        assert [event["event"] for event in feed.poll()][0] == jobs.EV_QUEUED

    def test_a_cursor_resumes_without_duplicates_and_without_a_gap(self, written):
        identifier = api.submit_minimax("one")
        first = api.subscribe(identifier)
        early = first.poll()
        jobs.drain_once()

        resumed = api.subscribe(identifier, cursor=early[-1]["seq"])
        rest = resumed.poll()

        assert [event["seq"] for event in early] == [1]
        assert [event["seq"] for event in rest] == [2, 3, 4, 5, 6]

    def test_a_feed_on_one_request_closes_when_that_request_ends(self, written):
        identifier = api.submit_minimax("one")
        feed = api.subscribe(identifier)
        jobs.drain_once()
        feed.poll()

        assert feed.closed is True
        assert feed.identifier not in [entry["feed"] for entry in jobs.feeds()]

    def test_events_stops_yielding_once_the_feed_closes(self, written):
        identifier = api.submit_minimax("one")
        feed = api.subscribe(identifier)
        jobs.drain_once()

        delivered = list(feed.events(timeout=2.0))

        assert delivered[-1]["event"] == jobs.EV_DONE

    def test_events_returns_at_its_timeout_when_nothing_happens(self, written):
        identifier = api.submit_minimax("one")
        feed = api.subscribe(identifier)
        feed.poll()
        started = time.monotonic()

        delivered = list(feed.events(timeout=0.2, idle=0.05))

        assert delivered == []
        assert time.monotonic() - started < 3.0

    def test_a_queue_wide_feed_sees_every_request_in_the_order_things_happened(self,
                                                                               written):
        feed = api.subscribe()
        first = api.submit_minimax("one")
        second = api.submit_minimax("two")
        while jobs.drain_once():
            pass

        seen = [(event["id"], event["event"]) for event in feed.poll()]

        assert (first, jobs.EV_QUEUED) in seen
        assert (second, jobs.EV_QUEUED) in seen
        assert seen.index((first, jobs.EV_DONE)) < seen.index((second, jobs.EV_STARTED))

    def test_a_queue_wide_feed_does_not_close_on_one_requests_terminal_event(self,
                                                                             written):
        feed = api.subscribe()
        api.submit_minimax("one")
        jobs.drain_once()
        feed.poll()

        assert feed.closed is False

    def test_subscribing_to_an_id_that_does_not_exist_is_refused_now(self):
        with pytest.raises(api.Rejected) as raised:
            api.subscribe("never-existed")

        assert raised.value.code == "unknown_job"

    def test_a_feed_nobody_reads_expires(self, written):
        identifier = api.submit_minimax("one")
        feed = api.subscribe(identifier, ttl=0.0)

        assert jobs.feeds() == []
        assert feed.expired is True

    def test_closing_a_feed_takes_it_out_of_the_register(self, written):
        feed = api.subscribe(api.submit_minimax("one"))

        feed.close()

        assert jobs.feeds() == []

    def test_forgetting_a_job_closes_the_feeds_watching_it(self, written):
        identifier = api.submit_minimax("one")
        feed = api.subscribe(identifier, cursor=99)
        jobs.drain_once()

        api.forget(identifier)

        assert feed.closed is True

    def test_a_long_run_drops_chunks_rather_than_its_terminal_event(self, monkeypatch):
        """The log is bounded, and what it sheds is the typewriter effect only."""
        monkeypatch.setattr(jobs, "MAX_EVENTS", 12)
        monkeypatch.setattr(sessions, "minimax", enhancement(chunks=40))
        identifier = api.submit_minimax("one")
        feed = api.subscribe(identifier)
        jobs.drain_once()

        names = [event["event"] for event in feed.poll()]

        assert names[0] == jobs.EV_QUEUED
        assert names[-1] == jobs.EV_DONE
        assert names.count(jobs.EV_CHUNK) < 40
        assert api.status(identifier)["dropped_events"] > 0
        assert api.status(identifier)["prompt"] == "Shot one."


# --------------------------------------------------------------------------- #
# History
# --------------------------------------------------------------------------- #


class TestTheHistory:
    """T7 — "as if somebody had gone to LLM Studio and asked for it"."""

    def test_a_finished_request_appears_in_saved_prompts(self, written):
        api.submit_minimax("a car chase", variant=enhancer.REF2VA, origin="video-tools")
        jobs.drain_once()

        saved = mc_llm_state.minimax_sessions()

        assert [entry.result for entry in saved] == ["Shot one."]
        assert saved[0].variant == enhancer.REF2VA
        assert saved[0].prompt == "a car chase"

    def test_bulk_callers_can_stay_out_of_somebody_elses_history(self, written):
        api.submit_minimax("a car chase", remember=False)
        jobs.drain_once()

        assert mc_llm_state.minimax_sessions() == []

    def test_a_failed_request_is_not_filed_as_a_prompt(self, monkeypatch):
        monkeypatch.setattr(sessions, "minimax", enhancement(fail="no model"))
        api.submit_minimax("a car chase")
        jobs.drain_once()

        assert mc_llm_state.minimax_sessions() == []


# --------------------------------------------------------------------------- #
# Capabilities
# --------------------------------------------------------------------------- #


class TestAskingWhatIsPossible:
    """T8 — a caller can find out before it asks, and is told why not."""

    def test_the_contract_describes_itself(self):
        found = api.capabilities()

        assert found["api_version"] == api.API_VERSION
        assert set(found["variants"]) == {enhancer.FL2VA, enhancer.REF2VA}
        assert found["slots"] == list(api.SLOTS)
        assert set(found["events"]) == set(jobs.EVENTS)

    def test_it_names_the_model_that_would_answer(self, monkeypatch):
        """The file's name and never its path: a path is somebody's home directory."""
        class Loaded:
            configured, sees = True, True
            model = "/home/someone/models/qwen3-vl-8b.gguf"

        import mc_llm_runtime

        monkeypatch.setattr(mc_llm_runtime, "config", lambda *a, **k: Loaded())
        found = api.capabilities()

        assert found["model"] == "qwen3-vl-8b.gguf"
        assert found["reason"] == ""

    def test_a_model_with_no_projector_is_reported_before_a_picture_is_sent(self,
                                                                            monkeypatch):
        class Blind:
            configured, sees, model = True, False, "/models/text.gguf"

        import mc_llm_runtime

        monkeypatch.setattr(mc_llm_runtime, "config", lambda *a, **k: Blind())
        found = api.capabilities()

        assert found["vision"] is False
        assert "vision projector" in found["reason"]

    def test_no_model_at_all_is_reported_as_that_and_not_as_blindness(self, monkeypatch):
        class Absent:
            configured, sees, model = False, False, ""

        import mc_llm_runtime

        monkeypatch.setattr(mc_llm_runtime, "config", lambda *a, **k: Absent())

        assert "Setup" in api.capabilities()["reason"]

    def test_a_runtime_that_cannot_be_read_does_not_take_the_call_down(self, monkeypatch):
        import mc_llm_runtime

        def broken(*args, **kwargs):
            raise RuntimeError("nothing is installed")

        monkeypatch.setattr(mc_llm_runtime, "config", broken)

        assert "nothing is installed" in api.capabilities()["reason"]


# --------------------------------------------------------------------------- #
# The panel
# --------------------------------------------------------------------------- #


class TestThePanelIsBlocked:
    """T9 — going to MiniMax H3 during an external request finds it inert."""

    def test_the_gate_is_open_when_nothing_external_is_happening(self):
        banner, actions, enhance = mc_llm_minimax_panel._gate()

        assert banner["visible"] is False
        assert actions["visible"] is False
        assert enhance["interactive"] is True

    def test_the_gate_closes_while_something_is_queued(self, written):
        api.submit_minimax("one")

        banner, actions, enhance = mc_llm_minimax_panel._gate()

        assert banner["visible"] is True
        assert actions["visible"] is True
        assert enhance["interactive"] is False

    def test_the_banner_names_who_is_asking_and_how_many_are_waiting(self, monkeypatch):
        import threading

        started, release = threading.Event(), threading.Event()
        monkeypatch.setattr(sessions, "minimax", blocking_enhancement(started, release))
        identifier = api.submit_minimax("one", origin="video-tools")
        api.submit_minimax("two")
        worker = threading.Thread(target=jobs.drain_once, daemon=True)
        worker.start()
        assert started.wait(timeout=5)

        said = mc_llm_minimax_panel._banner()

        assert "video-tools" in said
        assert identifier in said
        assert "1 more waiting" in said
        release.set()
        worker.join(timeout=5)

    def test_enhance_refuses_even_if_the_banner_has_not_caught_up(self, written):
        """The gate is a picture; this is the moment that matters."""
        api.submit_minimax("one")

        events = list(mc_llm_minimax_panel._enhance("my own prompt", "fl2va", None, 7))

        assert len(events) == 1
        assert "another extension" in events[0][3]
        assert events[0][4]["interactive"] is False

    def test_the_panel_can_stop_the_running_request(self, monkeypatch):
        import threading

        started, release = threading.Event(), threading.Event()
        monkeypatch.setattr(sessions, "minimax", blocking_enhancement(started, release))
        identifier = api.submit_minimax("one")
        worker = threading.Thread(target=jobs.drain_once, daemon=True)
        worker.start()
        assert started.wait(timeout=5)

        _banner, _actions, _enhance, status = mc_llm_minimax_panel._stop_external()

        assert identifier in status
        worker.join(timeout=5)
        assert api.status(identifier)["state"] == jobs.CANCELLED

    def test_stopping_with_nothing_running_says_so_rather_than_failing(self):
        *_updates, status = mc_llm_minimax_panel._stop_external()

        assert "Nothing external is running" in status

    def test_the_panel_can_clear_the_whole_queue(self, written):
        for index in range(3):
            api.submit_minimax(f"prompt {index}")

        *updates, status = mc_llm_minimax_panel._clear_external()

        assert "Cancelled 3 external requests" in status
        assert updates[2]["interactive"] is True

    def test_clearing_an_empty_queue_says_so(self):
        *_updates, status = mc_llm_minimax_panel._clear_external()

        assert "nothing external to cancel" in status

    def test_the_gate_survives_a_queue_it_cannot_read(self, monkeypatch):
        """A gate that raised would take the whole tab down to report a queue length."""
        def broken(*args, **kwargs):
            raise RuntimeError("gone")

        monkeypatch.setattr(jobs, "active", broken)
        monkeypatch.setattr(jobs, "snapshot", broken)

        assert mc_llm_minimax_panel.jobs_active() is False
        assert "Check again" in mc_llm_minimax_panel._banner()

    def test_the_gate_is_re_read_when_the_workspace_is_opened(self, written):
        built = mc_llm_minimax_panel.build()
        api.submit_minimax("one")

        updates = built["on_mode"]("minimax")

        assert updates[2]["interactive"] is False
        assert len(updates) == len(built["gate"])

    def test_the_timer_only_ticks_on_the_workspace_it_belongs_to(self, written):
        built = mc_llm_minimax_panel.build()

        on_minimax = built["on_mode"]("minimax")
        elsewhere = built["on_mode"]("chat")

        assert on_minimax[-1]["active"] is True
        assert elsewhere[-1]["active"] is False

    def test_a_host_without_a_timer_still_builds_a_working_gate(self, monkeypatch,
                                                                written):
        """``gr.Timer`` arrived in Gradio 4.30 and this extension takes what it is given."""
        import gradio

        monkeypatch.delattr(gradio, "Timer")
        built = mc_llm_minimax_panel.build()

        assert len(built["gate"]) == 3
        assert len(built["on_mode"]("minimax")) == 3
