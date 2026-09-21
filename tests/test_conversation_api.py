"""The conversation routes: the gates, and the feed's promises.

The handlers are tested as plain functions and the gates as plain functions,
because that is what they are -- the FastAPI wrappers do nothing but call them
and turn the answer into a response. What matters is what cannot get past them:
a page on another origin, a page holding a key from a process that has
restarted, a request for a picture by a path rather than by a ticket.

The feed's own promises are the other half. A cursor is only worth having if a
subscriber that fell behind can tell that it did, and if an event that lands
between reading the conversation and starting to listen is still delivered.
"""

from __future__ import annotations

import pytest

import mc_llm_attachment_staging as staging
import mc_llm_attachments
import mc_llm_conversation_api as api
import mc_llm_conversation_feed as feed
import mc_llm_conversation_ops as ops
import mc_llm_conversation_service as service
import mc_llm_conversation_store as store_module
import mc_llm_paths
from mc_llm_conversation_store import Key


@pytest.fixture(autouse=True)
def store(tmp_path, monkeypatch, host):
    monkeypatch.setattr(mc_llm_paths, "data_root", lambda: tmp_path)
    service.reset()
    store_module.registry.reset()
    ops.reset()
    feed.reset()
    staging.reset()
    mc_llm_attachments.forget_tickets()
    yield tmp_path
    service.reset()
    store_module.registry.reset()
    ops.reset()
    feed.reset()
    staging.reset()
    mc_llm_attachments.forget_tickets()


class Request:
    """As much of a request as the gates read, and nothing else."""

    def __init__(self, headers=None):
        self.headers = headers or {}


class TestTheCapability:
    def test_a_request_with_the_right_key_passes(self):
        api.checked(Request({api.HEADER: api.session_token(), "host": "forge.example"}))

    def test_a_request_with_no_key_is_refused(self):
        with pytest.raises(api.Refused) as refusal:
            api.checked(Request({"host": "forge.example"}))

        assert refusal.value.status == 401

    def test_a_key_from_a_process_that_has_gone_is_refused(self):
        stale = api.session_token()
        api.reset_token()

        with pytest.raises(api.Refused) as refusal:
            api.checked(Request({api.HEADER: stale}))

        assert refusal.value.status == 401

    def test_a_key_with_characters_compare_digest_dislikes_is_refused(self):
        """``compare_digest`` on ``str`` raises unless both sides are ASCII, so
        a header with one accented character used to leave a route as a 500
        rather than as the refusal it is."""
        with pytest.raises(api.Refused):
            api.checked(Request({api.HEADER: "café"}))

    def test_a_page_on_another_origin_is_refused(self):
        with pytest.raises(api.Refused) as refusal:
            api.checked(Request({api.HEADER: api.session_token(),
                                 "origin": "https://elsewhere.example",
                                 "host": "forge.example"}))

        assert refusal.value.status == 403

    def test_a_same_origin_request_passes(self):
        api.checked(Request({api.HEADER: api.session_token(),
                             "origin": "https://forge.example",
                             "host": "forge.example"}))

    def test_a_reverse_proxy_host_is_honoured_but_compared(self):
        api.checked(Request({api.HEADER: api.session_token(),
                             "origin": "https://public.example",
                             "host": "internal:7860",
                             "x-forwarded-host": "public.example"}))

        with pytest.raises(api.Refused):
            api.checked(Request({api.HEADER: api.session_token(),
                                 "origin": "https://attacker.example",
                                 "host": "internal:7860",
                                 "x-forwarded-host": "public.example"}))

    def test_a_missing_origin_is_not_treated_as_hostile(self):
        """Some browsers send no ``Origin`` on a same-origin fetch, and treating
        its absence as hostile would break the ordinary case to defend against
        one the key already covers."""
        api.checked(Request({api.HEADER: api.session_token(), "host": "forge.example"}))


class TestBootstrap:
    def test_it_says_what_this_server_can_do(self):
        found = api.bootstrap("page-1")

        assert found["protocol_version"] == service.PROTOCOL_VERSION
        assert found["server_epoch"] == service.SERVER_EPOCH
        assert "send" in found["actions"]
        assert "vision" in found["capabilities"]

    def test_it_never_carries_the_key(self):
        """The capability is put in the page by Python, not handed back by a
        route -- a route that returned it would be a route that could be asked
        for it."""
        assert api.session_token() not in repr(api.bootstrap("page-1"))


class TestCommands:
    def test_a_refusal_carries_the_status_it_maps_to(self, store):
        body = {"protocol_version": 99, "operation_id": "op-1", "action": "send",
                "conversation": {"character": "Ada", "thread_id": "x"},
                "expected_revision": None}

        payload, status = api.commands(body)

        assert payload["ok"] is False
        assert status == 400

    def test_a_conflict_is_a_409(self, store):
        from prompt_master.chat.history import ChatStore, USER

        chats = ChatStore(store / "chats")
        found = chats.new("Ada")
        found.append(USER, "ask")
        chats.save(found)
        body = {"protocol_version": service.PROTOCOL_VERSION,
                "server_epoch": service.SERVER_EPOCH, "operation_id": "op-1",
                "action": "delete_message",
                "conversation": {"character": "Ada", "thread_id": found.identifier},
                "expected_revision": {"kind": "revision", "value": 77},
                "target": {"index": 0}}

        payload, status = api.commands(body)

        assert payload["error"]["code"] == service.STALE_REVISION
        assert status == 409


class TestOperations:
    def test_an_unknown_operation_is_a_404_rather_than_a_guess(self):
        """The answer to "did my message send?" is never "probably not, try
        again" -- that is the answer that produces two messages."""
        payload, status = api.operation("nothing-was-ever-called-this")

        assert status == 404
        assert payload["error"]["code"] == service.NOT_FOUND


class TestAttachments:
    def test_an_upload_that_is_not_a_picture_is_a_422(self):
        payload, status = api.stage_upload(b"just some words", "x.png")

        assert status == 422
        assert payload["ok"] is False

    def test_an_upload_beyond_the_route_limit_is_a_413(self):
        payload, status = api.stage_upload(b"\0" * (api.MAX_UPLOAD + 1), "x.png")

        assert status == 413

    def test_a_staged_picture_comes_back_ready(self):
        import io

        from PIL import Image

        buffer = io.BytesIO()
        Image.new("RGB", (8, 8), (10, 20, 30)).save(buffer, format="PNG")

        payload, status = api.stage_upload(buffer.getvalue(), "x.png")

        assert status == 200
        assert payload["state"] == staging.READY
        assert staging.resolve(payload["token"]) is not None


class TestServingPictures:
    def test_a_ticket_is_not_a_path(self, store):
        """6.6. No filesystem path leaves the server -- a route that took one
        would be a route somebody could hand a different one."""
        from PIL import Image

        record = mc_llm_attachments.store(Image.new("RGB", (8, 8), (1, 2, 3)), "Ada")

        url = mc_llm_attachments.serving_url(record)

        assert url.startswith(mc_llm_attachments.ATTACHMENT_ROUTE)
        assert record not in url
        assert str(store) not in url

    def test_the_same_record_reuses_one_ticket(self, store):
        from PIL import Image

        record = mc_llm_attachments.store(Image.new("RGB", (8, 8), (1, 2, 3)), "Ada")

        assert mc_llm_attachments.serving_url(record) \
            == mc_llm_attachments.serving_url(record)

    def test_a_ticket_resolves_only_inside_the_attachment_folder(self, store):
        from PIL import Image

        record = mc_llm_attachments.store(Image.new("RGB", (8, 8), (1, 2, 3)), "Ada")
        url = mc_llm_attachments.serving_url(record)
        ticket = url.rsplit("/", 1)[-1]

        found = mc_llm_attachments.serving_path(ticket)

        assert found is not None
        assert mc_llm_attachments.root().resolve() in found.resolve().parents

    def test_a_ticket_nobody_minted_resolves_to_nothing(self):
        assert mc_llm_attachments.serving_path("made-up") is None
        assert mc_llm_attachments.serving_path("") is None

    def test_a_record_that_escapes_gets_no_ticket(self):
        assert mc_llm_attachments.serving_url("../../etc/passwd") == ""


class TestTheFeed:
    def test_a_subscriber_reads_what_happened_after_it_subscribed(self):
        page = feed.subscribe("page-1")
        service.publish(service.CONVERSATION_CHANGED, Key("Ada", "t1"), revision=3)

        events, gap = page.since(page.cursor)

        assert [event["kind"] for event in events] == [service.CONVERSATION_CHANGED]
        assert gap is False
        assert events[0]["conversation_revision"] == 3

    def test_an_event_between_the_snapshot_and_the_subscription_is_not_lost(self):
        """T01. The cursor is taken under the same lock the event is published
        under, so there is no window where an event belongs to nobody."""
        page = feed.subscribe("page-1")
        taken = page.cursor
        service.publish(service.CONVERSATION_CHANGED, Key("Ada", "t1"))

        events, _ = page.since(taken)

        assert len(events) == 1

    def test_a_cursor_that_has_been_trimmed_away_is_reported_as_a_gap(self):
        """Nothing is guessed. The page asks for a snapshot and starts again
        from a position both sides agree on."""
        page = feed.subscribe("page-1")
        for _ in range(3):
            service.publish(service.CONVERSATION_CHANGED, Key("Ada", "t1"))
        page._events = page._events[-1:]

        events, gap = page.since(0)

        assert gap is True

    def test_reply_patches_are_coalesced_and_the_terminal_never_is(self):
        """A thousand one-word events would push the start of the reply out of
        the ring. The patch carries the cumulative text, so dropping one is
        genuinely lossless."""
        page = feed.subscribe("page-1")
        for word in ("one", "one two", "one two three"):
            feed.publish(service.REPLY_PATCH, Key("Ada", "t1"), operation_id="op-1",
                         text=word)
        feed.publish(service.OPERATION_TERMINAL, Key("Ada", "t1"), operation_id="op-1",
                     force=True, text="one two three", phase="completed")

        events, _ = page.since(page.cursor)
        kinds = [event["kind"] for event in events]

        assert kinds.count(service.REPLY_PATCH) == 1, kinds
        assert kinds[-1] == service.OPERATION_TERMINAL
        assert events[-1]["payload"]["text"] == "one two three"

    def test_two_pages_each_get_their_own_copy(self):
        first = feed.subscribe("page-1")
        second = feed.subscribe("page-2")
        service.publish(service.CONVERSATION_CHANGED, Key("Ada", "t1"))

        assert len(first.poll()) == 1
        assert len(second.poll()) == 1
        assert first.poll() == []

    def test_a_feed_nobody_reads_expires(self):
        page = feed.subscribe("page-1", ttl=0.0)

        assert feed.expire() == 1
        assert feed.find(page.identifier) is None

    def test_a_ring_is_bounded(self):
        page = feed.subscribe("page-1")
        for index in range(feed.MAX_EVENTS + 50):
            feed.publish(service.CONVERSATION_CHANGED, Key("Ada", "t1"), index=index)

        assert len(page._events) <= feed.MAX_EVENTS

    def test_the_cursor_only_ever_goes_up(self):
        feed.subscribe("page-1")
        first = feed.publish(service.CONVERSATION_CHANGED, Key("Ada", "t1"))
        second = feed.publish(service.CONVERSATION_CHANGED, Key("Ada", "t1"))

        assert second > first

    def test_an_event_carries_the_epoch_so_a_restart_is_visible(self):
        page = feed.subscribe("page-1")
        service.publish(service.CONVERSATION_CHANGED, Key("Ada", "t1"))

        assert page.poll()[0]["server_epoch"] == service.SERVER_EPOCH


class TestWorkspaces:
    def test_it_describes_the_tab_this_extension_owns(self, monkeypatch):
        found = api.workspaces()

        assert found["ok"] is True
        assert isinstance(found["workspaces"], list)

    def test_a_disabled_tab_is_not_offered(self, monkeypatch):
        import mc_llm_studio

        monkeypatch.setattr(mc_llm_studio, "enabled", lambda: False)

        assert api.workspaces()["workspaces"] == []
