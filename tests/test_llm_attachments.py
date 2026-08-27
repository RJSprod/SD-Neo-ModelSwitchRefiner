"""A conversation's pictures: on disk, findable, and shown where they belong.

A chat used to carry its attachments inside itself, base64-encoded, in the same
JSON file as the words. What that cost was reported from use rather than
guessed at: the transcript is streamed to the browser on every token, so a
picture inside a message is re-sent on every token -- which is why the
transcript only ever showed *the name* of the picture, and why reopening a chat
a week later showed a line of italic text where a photograph had been.

So the bytes go into a folder somebody can open, and the conversation keeps a
path. These tests are mostly about the properties that makes possible: the same
picture stored twice is one file, a record cannot point outside the folder, and
a chat written before the folder existed is moved into it the first time it is
opened rather than by a script somebody has to know to run.
"""

from __future__ import annotations

import base64
import json

import pytest

import mc_llm_attachments as attachments
import mc_llm_chat_panel
import mc_llm_paths


@pytest.fixture(autouse=True)
def root(tmp_path, monkeypatch, host):
    monkeypatch.setattr(mc_llm_paths, "data_root", lambda: tmp_path)
    return tmp_path


def picture(colour=(200, 30, 30), size=(24, 18)):
    from PIL import Image

    return Image.new("RGB", size, colour)


def inline(colour=(10, 90, 200)) -> str:
    from prompt_master.imaging.preprocess import encode

    return encode(picture(colour))


class TestWhereTheyGo:
    def test_a_picture_lands_in_the_character_s_own_folder(self, root):
        record = attachments.store(picture(), "Ada")

        found = attachments.locate(record)
        assert found is not None and found.is_file()
        assert found.parent == root / attachments.DIRNAME / "Ada"

    def test_the_record_is_relative_so_the_install_can_move(self, root):
        """The same rule the state file follows for a model: an installation
        copied to another drive keeps its conversations whole."""
        record = attachments.store(picture(), "Ada")

        assert record == f"Ada/{record.split('/')[-1]}"
        assert not record.startswith("/") and ":" not in record

    def test_the_same_picture_twice_is_one_file(self, root):
        """The name is the hash of the bytes. A branch shares its parent's
        pictures rather than copying them, and an edit that re-attaches the
        same photograph writes nothing."""
        first = attachments.store(picture(), "Ada")
        second = attachments.store(picture(), "Ada")

        assert first == second
        assert len(list((root / attachments.DIRNAME / "Ada").glob("*.jpg"))) == 1

    def test_two_different_pictures_are_two_files(self, root):
        first = attachments.store(picture((200, 30, 30)), "Ada")
        second = attachments.store(picture((30, 200, 30)), "Ada")

        assert first != second

    def test_a_character_whose_name_is_not_a_filename_still_gets_a_folder(self, root):
        record = attachments.store(picture(), "Ada / Lovelace?")

        assert attachments.locate(record) is not None

    def test_nothing_is_left_half_written(self, root):
        """Written beside and renamed. A file named after what its contents
        hash to must never be able to hold something else."""
        attachments.store(picture(), "Ada")

        assert list((root / attachments.DIRNAME / "Ada").glob("*.part")) == []


class TestFindingThemAgain:
    def test_a_record_whose_file_has_gone_finds_nothing(self, root):
        record = attachments.store(picture(), "Ada")
        attachments.locate(record).unlink()

        assert attachments.locate(record) is None

    def test_a_record_cannot_point_outside_the_folder(self, root):
        """A record comes out of a JSON file, and a JSON file is a thing
        somebody can edit."""
        secret = root / "setup-state.json"
        secret.write_text("{}", encoding="utf-8")

        assert attachments.locate("../setup-state.json") is None

    def test_nothing_recorded_finds_nothing(self):
        assert attachments.locate("") is None

    def test_a_stored_picture_reads_back_as_the_url_inference_carries(self, root):
        record = attachments.store(picture(), "Ada")

        assert attachments.data_url(record).startswith("data:image/jpeg;base64,")

    def test_a_picture_that_has_gone_reads_back_as_nothing(self, root):
        record = attachments.store(picture(), "Ada")
        attachments.locate(record).unlink()

        assert attachments.data_url(record) == ""


class TestShowingThem:
    def test_the_markup_points_at_the_file_rather_than_carrying_it(self, root):
        """The whole reason there is a folder. The transcript is re-sent on
        every token of a reply, so what is in it has to be a path."""
        record = attachments.store(picture(), "Ada")

        shown = attachments.markup(record, "frame.png")

        assert shown.startswith("<img src=\"file=")
        assert "base64" not in shown
        assert 'alt="frame.png"' in shown

    def test_a_name_with_markup_in_it_is_escaped(self, root):
        record = attachments.store(picture(), "Ada")

        shown = attachments.markup(record, '"><script>alert(1)</script>')

        assert "<script>" not in shown

    def test_a_picture_that_has_gone_says_so_rather_than_vanishing(self, root):
        """A message that was sent with a picture is not the same message
        without one, and the reply underneath is about something the reader can
        no longer see."""
        record = attachments.store(picture(), "Ada")
        attachments.locate(record).unlink()

        assert attachments.markup(record) == attachments.MISSING


class TestMovingAnOldChatIn:
    def thread(self, root, count=1):
        from prompt_master.chat.history import ASSISTANT, ChatStore, USER

        chats = ChatStore(root / "chats")
        conversation = chats.new("Ada")
        for index in range(count):
            conversation.append(USER, f"look {index}", inline((10 * index, 90, 200)),
                                f"frame{index}.png")
            conversation.append(ASSISTANT, f"I see {index}")
        chats.save(conversation)
        return chats, conversation

    def test_an_inline_picture_becomes_a_file(self, root):
        chats, conversation = self.thread(root)

        assert attachments.adopt(conversation, "Ada") is True
        assert attachments.locate(conversation.messages[0].image_path) is not None

    def test_the_chat_stops_carrying_it(self, root):
        chats, conversation = self.thread(root)

        attachments.adopt(conversation, "Ada")
        chats.save(conversation)

        written = json.loads(chats.path_for("Ada", conversation.identifier).read_text())
        assert "image" not in written["messages"][0]
        assert written["messages"][0]["image_path"]

    def test_the_bytes_are_the_bytes_that_were_sent(self, root):
        chats, conversation = self.thread(root)
        was = conversation.messages[0].image

        attachments.adopt(conversation, "Ada")

        kept = attachments.locate(conversation.messages[0].image_path).read_bytes()
        assert base64.b64encode(kept).decode() == was.split(",", 1)[1]

    def test_a_chat_with_nothing_to_move_is_not_touched(self, root):
        from prompt_master.chat.history import ChatStore, USER

        chats = ChatStore(root / "chats")
        conversation = chats.new("Ada")
        conversation.append(USER, "just words")

        assert attachments.adopt(conversation, "Ada") is False

    def test_a_picture_that_cannot_be_decoded_stays_where_it_is(self, root):
        """Losing an attachment to a tidying-up would be far worse than a chat
        that goes on carrying one."""
        from prompt_master.chat.history import ChatStore, USER

        chats = ChatStore(root / "chats")
        conversation = chats.new("Ada")
        conversation.append(USER, "look", "data:image/jpeg;base64,not base64 at all", "x.png")

        assert attachments.adopt(conversation, "Ada") is False
        assert conversation.messages[0].image

    def test_opening_a_thread_moves_it_without_being_asked(self, root):
        """The migration runs where a chat is read, not in a script somebody
        has to know about."""
        chats, conversation = self.thread(root, count=2)

        opened = mc_llm_chat_panel._load("Ada", conversation.identifier)

        assert all(message.image_path or not message.attached
                   for message in opened.messages)
        assert not any(message.image for message in opened.messages)

    def test_and_the_move_is_written_down(self, root):
        chats, conversation = self.thread(root)

        mc_llm_chat_panel._load("Ada", conversation.identifier)
        reopened = chats.load("Ada", conversation.identifier)

        assert reopened.messages[0].image_path
        assert not reopened.messages[0].image


class TestTheMessageKeepsOneOrTheOther:
    def test_a_stored_picture_is_written_as_a_path(self):
        from prompt_master.chat.history import Message

        written = Message(role="user", image_path="Ada/abc.jpg").to_dict()

        assert written["image_path"] == "Ada/abc.jpg"
        assert "image" not in written

    def test_a_message_that_never_moved_keeps_its_inline_copy(self):
        from prompt_master.chat.history import Message

        written = Message(role="user", image="data:image/jpeg;base64,AA").to_dict()

        assert written["image"] == "data:image/jpeg;base64,AA"
        assert "image_path" not in written

    def test_a_message_with_no_picture_writes_neither(self):
        from prompt_master.chat.history import Message

        written = Message(role="user").to_dict()

        assert "image" not in written and "image_path" not in written

    def test_attached_answers_for_both(self):
        from prompt_master.chat.history import Message

        assert Message(role="user", image_path="Ada/a.jpg").attached
        assert Message(role="user", image="data:image/jpeg;base64,AA").attached
        assert not Message(role="user").attached

    def test_a_chat_written_before_any_of_this_still_reads(self):
        from prompt_master.chat.history import Message

        found = Message.from_dict({"role": "user", "versions": ["look"],
                                   "image": "data:image/jpeg;base64,AA",
                                   "image_name": "frame.png"})

        assert found.image == "data:image/jpeg;base64,AA"
        assert found.image_path == ""


class TestTheRequestReadsThemBack:
    def test_the_picture_is_loaded_for_the_message_being_sent(self, root):
        from prompt_master.chat.history import Message

        record = attachments.store(picture(), "Ada")
        messages = [Message(role="user", versions=["look"], image_path=record)]

        mc_llm_chat_panel._with_pictures(messages)

        assert messages[0].image.startswith("data:image/jpeg;base64,")

    def test_only_as_many_as_a_request_can_carry_are_read(self, root):
        """The builder keeps at most MAX_IMAGES of the stills that survive
        trimming. Decoding forty photographs to send four would be forty reads
        a message."""
        from prompt_master.chat.history import Message
        from prompt_master.chat.prompt import MAX_IMAGES

        messages = [Message(role="user", versions=[f"look {index}"],
                            image_path=attachments.store(picture((index, 90, 200)), "Ada"))
                    for index in range(MAX_IMAGES + 3)]

        mc_llm_chat_panel._with_pictures(messages)

        assert sum(bool(message.image) for message in messages) == MAX_IMAGES
        # The newest, because those are the ones a request keeps.
        assert all(message.image for message in messages[-MAX_IMAGES:])

    def test_what_was_read_is_never_written_back(self, root):
        """The conversation on disk holds paths. A chat that saved the decoded
        copy beside the path would be exactly as large as it was before any of
        this."""
        from prompt_master.chat.history import Message

        record = attachments.store(picture(), "Ada")
        message = Message(role="user", versions=["look"], image_path=record)

        mc_llm_chat_panel._with_pictures([message])

        assert "image" not in message.to_dict()
