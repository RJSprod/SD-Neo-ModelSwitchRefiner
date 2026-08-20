"""Krea 2: the prompt package, the run, the panel and the history.

Almost everything here is about one thing, and it is not prompt quality --
which no test can assert -- but *reference identity*. The user says "the woman
from image 2"; four layers later a caption has to still be the second one, in
the second numbered line, under an instruction that has told the model not to
swap them. Every layer in between is a place that ordering can be lost:
a Gradio control that reorders on delete, a captioning loop that runs
concurrently, a synthesis message that hands over unlabelled paragraphs, a
history file that stores the pictures in whatever order a dict iterated.

So the tests below follow one number from the slot it was typed beside to the
sentence the writer is given, and the rest of them are about the two failures
that must never be papered over: a model that cannot see, and a reference that
could not be described.
"""

from __future__ import annotations

import pytest

import mc_broker
import mc_llm_krea_panel as panel
import mc_llm_paths
import mc_llm_sessions as sessions
import mc_llm_state as state
from prompt_master.krea import enhancer
from prompt_master.krea.references import KreaPromptResult, Reference


@pytest.fixture
def store(tmp_path, monkeypatch):
    """Point every LLM Studio file at a throwaway directory."""
    monkeypatch.setattr(mc_llm_paths, "data_root", lambda: tmp_path)
    yield tmp_path


class FakeClient:
    """A llama.cpp client that streams a canned answer, and remembers the asking.

    ``answers`` lets one run give a different reply per call, which is what the
    captioning tests need: four references means five calls, and the point of
    the test is that call two is about image 1 and call three about image 2.
    """

    def __init__(self, pieces=("Hello", " there"), fail=None, answers=None):
        self.pieces = list(pieces)
        self.fail = fail
        self.answers = list(answers or [])
        self.calls: list[dict] = []

    def stream_chat(self, messages, max_tokens, seed, on_text, cancel=None,
                    temperature=0.85, top_p=0.95, extra_sampling=None):
        self.calls.append({"messages": messages, "max_tokens": max_tokens, "seed": seed,
                           "temperature": temperature, "top_p": top_p,
                           "extra_sampling": dict(extra_sampling or {})})
        if self.fail is not None:
            raise self.fail
        if self.answers:
            answer = self.answers.pop(0)
            if callable(answer):
                answer = answer()
            on_text(answer)
            return answer
        produced = []
        for piece in self.pieces:
            if cancel is not None and cancel.is_set():
                break
            produced.append(piece)
            on_text(piece)
        return "".join(produced)


@pytest.fixture
def client(monkeypatch, host):
    """Install a fake client, record what vision was asked for, clear the register."""
    mc_broker.clear()
    monkeypatch.setattr(mc_broker, "host_busy", lambda: False)
    fake = FakeClient()
    asked: list[bool] = []

    def obtain(needs_vision=False):
        asked.append(bool(needs_vision))
        return fake

    monkeypatch.setattr(sessions, "_client", obtain)
    monkeypatch.setattr(sessions, "_placement_notes", list)
    fake.vision_asked = asked
    yield fake
    mc_broker.clear()


def drain(generator) -> list[sessions.Event]:
    return list(generator)


def kinds(events) -> list[str]:
    return [event.kind for event in events]


def texts(events, kind) -> list[str]:
    return [event.text for event in events if event.kind == kind]


def refs(count: int) -> list[Reference]:
    return [Reference(ui_index=position, path=f"/tmp/pic{position}.png",
                      data_url=f"data:image/png;base64,PIC{position}")
            for position in range(1, count + 1)]


# --------------------------------------------------------------------------- #
# The prompt package (design intent §3, §6, §7)
# --------------------------------------------------------------------------- #


class TestTheKreaInstruction:
    def test_the_base_instruction_is_kreas_own_file(self):
        """Read from expansion.txt rather than restated in Python, so that
        re-vendoring upstream is a copy and a digest rather than an edit."""
        from pathlib import Path

        import prompt_master.krea as package

        vendored = Path(package.__file__).with_name("expansion.txt")
        assert enhancer.base_instruction() == vendored.read_text(encoding="utf-8").strip()
        assert "expert prompt engineer for text-to-image models" in enhancer.base_instruction()

    def test_the_local_addendum_is_not_in_the_upstream_file(self):
        """§3: do not edit the upstream prompt text in place to add local
        behaviour. The two halves have to stay tellable apart."""
        assert enhancer.REFERENCE_ADDENDUM not in enhancer.base_instruction()
        assert "Image 1" not in enhancer.base_instruction()

    def test_a_text_only_system_message_is_exactly_upstreams(self):
        assert enhancer.system_prompt(False) == enhancer.base_instruction()

    def test_references_append_the_addendum_under_a_heading_that_names_it(self):
        assembled = enhancer.system_prompt(True)

        assert assembled.startswith(enhancer.base_instruction())
        assert enhancer.REFERENCE_HEADING in assembled
        assert enhancer.REFERENCE_ADDENDUM in assembled

    def test_the_addendum_forbids_the_failures_it_exists_to_prevent(self):
        """§6 is a list of ways a reference edit loses its meaning. Each one
        that is not stated is one the model has no reason not to do."""
        said = enhancer.REFERENCE_ADDENDUM.lower()

        assert "never swap" in said
        assert "authoritative" in said
        assert "style reference" in said
        assert "identity reference" in said
        assert "image 1" in said

    def test_the_upstream_rules_that_matter_are_still_in_the_instruction(self):
        """Not a test of Krea's writing -- a test that the vendored file is the
        one the design intent was written against, and has not been swapped for
        something shorter."""
        base = enhancer.base_instruction()

        for expected in ("Faithfulness First", "Text Rendering", "Respect Existing Detail",
                         "Preserve User Medium", "No bullets, JSON, or markdown"):
            assert expected in base


class TestTheMessagesTheModelIsGiven:
    def test_text_only_carries_no_reference_section(self):
        built = enhancer.messages("a woman walking through a rainy Tokyo street at night")

        assert built[0]["content"] == enhancer.base_instruction()
        assert "reference_images" not in built[1]["content"]
        assert built[1]["content"].startswith("user_prompt:")

    def test_the_user_s_wording_reaches_the_model_unaltered(self):
        """The prompt is written *about* what the user said. Tidying it first
        would mean writing about a paraphrase."""
        asked = 'A "GRAND OPENING" banner over a bakery — 35mm, shot on film.'

        content = enhancer.messages(asked)[1]["content"]

        assert asked in content

    def test_a_request_for_visible_text_keeps_its_quotes(self):
        content = enhancer.messages('a sign reading "OPEN 24 HOURS"')[1]["content"]

        assert '"OPEN 24 HOURS"' in content

    def test_references_are_numbered_and_labelled_in_the_user_turn(self):
        """§7: the writer is never handed unlabelled paragraphs and asked to
        work out which image is which."""
        content = enhancer.messages(
            "replace the face in image 1 with the woman from image 2",
            ["a woman on a balcony", "a woman in a red coat"])[1]["content"]

        assert "reference_images:" in content
        assert "Image 1: a woman on a balcony" in content
        assert "Image 2: a woman in a red coat" in content
        assert content.index("Image 1:") < content.index("Image 2:")

    def test_the_captions_keep_the_order_they_were_given_in(self):
        """The one property the whole feature rests on. Reversed captions are a
        reversed edit, and nothing downstream could tell."""
        forward = enhancer.reference_block(["first", "second", "third", "fourth"])

        assert forward.splitlines() == ["Image 1: first", "Image 2: second",
                                        "Image 3: third", "Image 4: fourth"]

    def test_a_reference_run_gets_the_addendum_and_a_text_run_does_not(self):
        with_refs = enhancer.messages("edit it", ["a portrait"])[0]["content"]
        without = enhancer.messages("edit it")[0]["content"]

        assert enhancer.REFERENCE_ADDENDUM in with_refs
        assert enhancer.REFERENCE_ADDENDUM not in without

    def test_the_captioner_is_told_to_describe_and_nothing_else(self):
        """§5: a captioner that starts editing produces a caption the writer
        then writes a prompt about, and the user's instruction is what loses."""
        said = enhancer.CAPTION_INSTRUCTION.lower()

        assert "do not name real people" in said
        assert "do not suggest how the image should be changed" in said
        assert "one paragraph" in said
        assert enhancer.CAPTION_TEMPERATURE == 0.0

    def test_the_captioner_is_shown_the_picture_and_the_instruction(self):
        built = enhancer.caption_messages("data:image/png;base64,AAAA")

        content = built[0]["content"]
        assert content[0]["image_url"]["url"] == "data:image/png;base64,AAAA"
        assert content[1]["text"] == enhancer.CAPTION_INSTRUCTION

    def test_the_captioner_is_never_told_what_the_edit_is(self):
        """It is describing a picture, not carrying out a request."""
        built = enhancer.caption_messages("data:image/png;base64,AAAA")

        assert all("user_prompt" not in str(part) for part in built[0]["content"])


class TestCleaningWhatComesBack:
    def test_a_thinking_block_is_removed(self):
        """Upstream's own instruction asks the model to think before it writes,
        so a visible think block is a model obeying a file this package must
        not edit -- it comes off here instead."""
        assert enhancer.clean("<think>weighing two styles</think>A quiet street.") == \
            "A quiet street."

    def test_an_enclosing_code_fence_is_removed(self):
        assert enhancer.clean("```\nA quiet street.\n```") == "A quiet street."
        assert enhancer.clean("```text\nA quiet street.\n```") == "A quiet street."

    def test_a_fence_in_the_middle_is_left_alone(self):
        """Only a single *enclosing* fence is contamination. Anything else is
        the prompt, and this function does not get to rewrite the prompt."""
        text = 'a photo of a screen showing ```code``` on it'

        assert enhancer.clean(text) == text

    def test_nothing_else_is_touched(self):
        """§7: formatting cleanup is allowed, semantic rewriting is not."""
        written = ('A photograph of a woman in a red coat crossing a rain-slick '
                   'Tokyo street at night, neon signage reading "OPEN" behind her.')

        assert enhancer.clean(written) == written

    def test_an_empty_answer_cleans_to_nothing_so_it_can_be_refused(self):
        assert enhancer.clean("   ") == ""
        assert enhancer.clean("<think>hm</think>") == ""
        assert enhancer.clean(None) == ""


class TestTheReferenceModel:
    def test_a_reference_is_numbered_the_way_the_user_sees_it(self):
        assert Reference(ui_index=2).label == "Image 2"

    def test_only_the_file_name_leaves_the_reference_never_the_path(self):
        """§14: a full path is somebody's home directory and often the name of
        the project they are working on."""
        assert Reference(ui_index=1, path="/home/someone/secret project/a.png").name == "a.png"

    def test_a_result_keeps_the_prompt_and_the_references_separable(self):
        """§12: do not reduce the whole task to one string. A backend adapter
        may reorder these; it may not redefine what Image 1 meant."""
        result = KreaPromptResult(prompt="a prompt", references=refs(2))

        assert result.prompt == "a prompt"
        assert result.names == ["pic1.png", "pic2.png"]
        assert [reference.ui_index for reference in result.references] == [1, 2]

    def test_a_role_is_never_guessed(self):
        """§13 rules out automatic reference role classification. The field
        exists for a role the user declared, and starts empty."""
        assert all(reference.semantic_role == "" for reference in refs(4))


# --------------------------------------------------------------------------- #
# The run (design intent §5, §8, §10)
# --------------------------------------------------------------------------- #


class TestTextOnlyRuns:
    def test_it_asks_for_a_client_that_does_not_need_to_see(self, client):
        """§8: text-only Krea prompting must work with any text-capable model."""
        drain(sessions.krea("a rainy Tokyo street", [], 7, sessions.Cancellation()))

        assert client.vision_asked == [False]

    def test_no_caption_event_is_emitted_and_one_request_is_made(self, client):
        events = drain(sessions.krea("a rainy Tokyo street", [], 7, sessions.Cancellation()))

        assert sessions.CAPTION not in kinds(events)
        assert len(client.calls) == 1
        assert kinds(events)[-1] == sessions.DONE

    def test_the_text_streams_through_chunk_events_before_done(self, client):
        events = drain(sessions.krea("a shot", [], 7, sessions.Cancellation()))

        assert texts(events, sessions.CHUNK) == ["Hello", " there"]
        assert texts(events, sessions.DONE) == ["Hello there"]

    def test_done_carries_the_cleaned_prompt(self, client, monkeypatch):
        monkeypatch.setattr(sessions, "_client", lambda needs_vision=False: FakeClient(
            pieces=("<think>weighing it up</think>", "A quiet street.")))

        events = drain(sessions.krea("a shot", [], 7, sessions.Cancellation()))

        assert texts(events, sessions.DONE) == ["A quiet street."]

    def test_an_empty_answer_is_an_error_not_an_empty_prompt(self, client, monkeypatch):
        monkeypatch.setattr(sessions, "_client",
                            lambda needs_vision=False: FakeClient(pieces=("",)))

        events = drain(sessions.krea("a shot", [], 7, sessions.Cancellation()))

        assert kinds(events)[-1] == sessions.FAILED


class TestReferenceRuns:
    def test_references_ask_for_a_client_that_can_see(self, client):
        drain(sessions.krea("edit it", refs(1), 7, sessions.Cancellation()))

        assert client.vision_asked == [True]

    def test_one_caption_event_per_reference_in_slot_order(self, client, monkeypatch):
        fake = FakeClient(answers=["a woman on a balcony", "a woman in a red coat",
                                   "a snowy forest", "warm side light", "the final prompt"])
        monkeypatch.setattr(sessions, "_client", lambda needs_vision=False: fake)

        events = drain(sessions.krea("use them", refs(4), 7, sessions.Cancellation()))

        assert texts(events, sessions.CAPTION) == [
            "a woman on a balcony", "a woman in a red coat", "a snowy forest",
            "warm side light"]
        assert kinds(events)[-1] == sessions.DONE

    def test_each_reference_is_captioned_sequentially_and_in_order(self, client):
        """Sequential is not an implementation detail here: it is what makes the
        Nth caption event the Nth image without either side carrying an index."""
        drain(sessions.krea("use them", refs(3), 7, sessions.Cancellation()))

        shown = [call["messages"][0]["content"][0]["image_url"]["url"]
                 for call in client.calls[:3]]
        assert shown == ["data:image/png;base64,PIC1", "data:image/png;base64,PIC2",
                         "data:image/png;base64,PIC3"]

    def test_the_writer_is_given_the_captions_under_their_own_numbers(self, client,
                                                                      monkeypatch):
        fake = FakeClient(answers=["a woman on a balcony", "a woman in a red coat",
                                   "the final prompt"])
        monkeypatch.setattr(sessions, "_client", lambda needs_vision=False: fake)

        drain(sessions.krea("replace the face in image 1 with the woman from image 2",
                            refs(2), 7, sessions.Cancellation()))

        written = fake.calls[-1]["messages"]
        assert enhancer.REFERENCE_ADDENDUM in written[0]["content"]
        assert "Image 1: a woman on a balcony" in written[1]["content"]
        assert "Image 2: a woman in a red coat" in written[1]["content"]

    def test_the_writing_pass_is_text_only_over_the_captions(self, client):
        """Caption-first, so no multi-image transport is needed anywhere."""
        drain(sessions.krea("use them", refs(2), 7, sessions.Cancellation()))

        assert isinstance(client.calls[-1]["messages"][1]["content"], str)

    def test_the_captions_are_kept_on_the_references_they_describe(self, client,
                                                                   monkeypatch):
        fake = FakeClient(answers=["first picture", "second picture", "the prompt"])
        monkeypatch.setattr(sessions, "_client", lambda needs_vision=False: fake)
        references = refs(2)

        drain(sessions.krea("use them", references, 7, sessions.Cancellation()))

        assert [reference.caption for reference in references] == ["first picture",
                                                                   "second picture"]

    def test_the_captioner_is_run_at_its_own_settings_not_the_writer_s(self, client):
        drain(sessions.krea("use it", refs(1), 7, sessions.Cancellation()))

        assert client.calls[0]["temperature"] == enhancer.CAPTION_TEMPERATURE
        assert client.calls[0]["max_tokens"] == enhancer.CAPTION_MAX_TOKENS
        assert client.calls[-1]["temperature"] == enhancer.TEMPERATURE


class TestWhenSomethingGoesWrong:
    def test_a_reference_that_cannot_be_described_stops_the_run(self, client, monkeypatch):
        """§8: do not synthesize a prompt from only the remaining images, and do
        not silently renumber the survivors."""
        fake = FakeClient(answers=["a woman on a balcony", "", "the final prompt"])
        monkeypatch.setattr(sessions, "_client", lambda needs_vision=False: fake)

        events = drain(sessions.krea("use them", refs(3), 7, sessions.Cancellation()))

        assert kinds(events)[-1] == sessions.FAILED
        assert len(texts(events, sessions.CAPTION)) == 1
        assert len(fake.calls) == 2, "the third image and the writer were never asked for"

    def test_the_failure_says_which_reference_it_was(self, client, monkeypatch):
        fake = FakeClient(answers=["first", "", "third", "the prompt"])
        monkeypatch.setattr(sessions, "_client", lambda needs_vision=False: fake)

        events = drain(sessions.krea("use them", refs(3), 7, sessions.Cancellation()))

        assert "image 2" in texts(events, sessions.FAILED)[0]

    def test_a_missing_projector_surfaces_as_a_failure_rather_than_a_text_prompt(
            self, client, monkeypatch):
        """The panel refuses first; this is the same requirement stated where
        the client is actually obtained, for anything that got past it."""
        def refuse(needs_vision=False):
            if needs_vision:
                raise RuntimeError("this request carries an image and the model running "
                                   "has no vision projector")
            return FakeClient()

        monkeypatch.setattr(sessions, "_client", refuse)

        events = drain(sessions.krea("use it", refs(1), 7, sessions.Cancellation()))

        assert kinds(events)[-1] == sessions.FAILED
        assert "vision projector" in texts(events, sessions.FAILED)[0]

    def test_a_cancel_during_captioning_stops_the_later_work(self, client, monkeypatch):
        cancel = sessions.Cancellation()
        fake = FakeClient(answers=[lambda: (cancel.cancel(), "first")[1],
                                   "second", "the prompt"])
        monkeypatch.setattr(sessions, "_client", lambda needs_vision=False: fake)

        events = drain(sessions.krea("use them", refs(2), 7, cancel))

        assert kinds(events)[-1] == sessions.CANCELLED
        assert len(fake.calls) == 1
        assert sessions.CAPTION not in kinds(events)

    def test_a_cancel_during_writing_stops_without_a_done_event(self, client):
        cancel = sessions.Cancellation()

        stream = sessions.krea("a shot", [], 7, cancel)
        first = next(stream)
        cancel.cancel()
        events = [first] + drain(stream)

        assert sessions.DONE not in kinds(events)
        assert kinds(events)[-1] == sessions.CANCELLED


class TestTheWorkloadIsAlwaysGivenBack:
    """§8 and §15: the lock and the GPU come back whatever happened, or every
    later run -- image generation included -- is stranded behind it."""

    def _free(self) -> bool:
        if mc_broker.active() is not None:
            return False
        with mc_broker.workload(mc_broker.FAMILY_IMAGE, "a pass", timeout=0.1) as held:
            return bool(held)

    def test_after_a_successful_run(self, client):
        drain(sessions.krea("a shot", [], 7, sessions.Cancellation()))

        assert self._free()

    def test_after_a_failed_run(self, client, monkeypatch):
        monkeypatch.setattr(sessions, "_client",
                            lambda needs_vision=False: FakeClient(fail=RuntimeError("boom")))

        events = drain(sessions.krea("a shot", [], 7, sessions.Cancellation()))

        assert kinds(events)[-1] == sessions.FAILED
        assert self._free()

    def test_after_a_cancelled_run(self, client, monkeypatch):
        cancel = sessions.Cancellation()
        fake = FakeClient(answers=[lambda: (cancel.cancel(), "first")[1], "the prompt"])
        monkeypatch.setattr(sessions, "_client", lambda needs_vision=False: fake)

        drain(sessions.krea("use it", refs(1), 7, cancel))

        assert self._free()

    def test_after_a_generator_that_was_abandoned_part_way(self, client):
        """A Gradio cancel closes the generator where it stands; the finally
        block is what gives the lock back."""
        stream = sessions.krea("a shot", [], 7, sessions.Cancellation())
        next(stream)
        stream.close()

        assert self._free()


class TestWhatTheConsoleIsTold:
    def test_the_run_is_named_and_counted_but_never_quoted(self, client, caplog):
        """§14: avoid logging complete private prompts, and never log an image."""
        with caplog.at_level("INFO", logger="model_chain"):
            drain(sessions.krea("a very private idea", refs(2), 7, sessions.Cancellation()))

        said = " ".join(record.getMessage() for record in caplog.records)
        assert "a Krea prompt from 2 references" in said
        assert "a very private idea" not in said
        assert "base64" not in said
        assert "PIC1" not in said


# --------------------------------------------------------------------------- #
# The panel (design intent §4, §8, §9)
# --------------------------------------------------------------------------- #


class TestThePanel:
    def test_it_builds_and_exposes_what_the_shell_needs(self, store, host):
        """A superset check, not an exact one: the shell wires these three and
        the panel is free to hand back more of itself than the shell reads."""
        built = panel.build()

        assert {"status", "output", "stop"} <= set(built)

    def test_the_output_box_does_not_grow_while_it_is_written(self, store, host):
        """A box that grows walks Stop off the bottom of the window at the one
        moment somebody wants to press it."""
        written = panel.build()["output"]

        assert written.max_lines == written.lines

    def test_an_empty_request_is_refused_before_the_runtime_is_touched(self, store, host):
        frames = list(panel._generate("   ", 7, 1))

        assert len(frames) == 1
        assert "Describe the image you want" in frames[0][3]

    def test_stop_gives_the_controls_back(self):
        """``cancels=`` closes the generator where it stands, so the handler
        that re-enables Generate has to be this one."""
        _, generate, stop = panel._cancel(None)

        assert generate.get("interactive") is True
        assert stop.get("interactive") is False

    def test_clear_empties_the_request_the_prompt_the_captions_and_the_slots(self):
        cleared = panel._clear()

        prompt, written, captions, _status = cleared[:4]
        slots = cleared[4:]
        assert prompt == "" and written == ""
        assert captions.get("value") == "" and captions.get("visible") is False
        assert slots == (None,) * enhancer.MAX_REFERENCES

    def test_there_is_one_slot_per_supported_reference(self, store, host):
        """Numbered slots rather than a multi-upload control: a Gradio file list
        reorders itself when an entry is deleted and replaced, and §4 says image
        identity may not come from upload order."""
        panel.build()

        assert enhancer.MAX_REFERENCES == 4

    def test_no_image_generation_control_is_offered(self):
        """§13: sampler, CFG, steps, LoRA strength, masks, style strength,
        moodboards and negative prompts belong to an image-generation
        integration, not to prompt authoring. Read out of the controls the
        panel declares rather than written down here, so a control added
        tomorrow is covered the moment it exists."""
        import ast
        from pathlib import Path

        tree = ast.parse(Path(panel.__file__).read_text(encoding="utf-8"))
        labelled = [keyword.value.value.lower()
                    for node in ast.walk(tree) if isinstance(node, ast.Call)
                    for keyword in node.keywords
                    if keyword.arg in ("label", "info", "placeholder")
                    and isinstance(keyword.value, ast.Constant)
                    and isinstance(keyword.value.value, str)]

        assert any("krea 2 prompt" in text for text in labelled), "the output box is there"
        for forbidden in ("cfg", "sampler", "step", "lora", "negative", "moodboard",
                          "mask", "strength", "resolution", "guidance"):
            assert not any(forbidden in text for text in labelled), forbidden


class TestTheNumberingContract:
    def test_the_visible_order_is_what_reaches_the_session(self):
        """§4: identity comes from slot order and from nothing else -- not the
        filename, not the upload time, not what the picture contains."""
        found, complaint = panel.references(["/tmp/zzz.png", "/tmp/aaa.png", None, None])

        assert complaint == ""
        assert [(r.ui_index, r.name) for r in found] == [(1, "zzz.png"), (2, "aaa.png")]

    def test_an_empty_workspace_has_no_references_and_no_complaint(self):
        assert panel.references([None, None, None, None]) == ([], "")

    def test_four_slots_number_one_to_four(self):
        found, _ = panel.references([f"/tmp/{n}.png" for n in range(1, 5)])

        assert [reference.ui_index for reference in found] == [1, 2, 3, 4]

    def test_a_gap_is_refused_rather_than_closed_up(self):
        """§8: do not silently renumber. The user has written "image 3" in their
        instruction and would get a prompt about a picture called Image 2."""
        found, complaint = panel.references(["/tmp/a.png", None, "/tmp/c.png", None])

        assert found == []
        assert "Image 2 is empty" in complaint

    def test_the_complaint_names_every_empty_slot(self):
        _, complaint = panel.references([None, None, None, "/tmp/d.png"])

        assert "Image 1" in complaint and "Image 2" in complaint and "Image 3" in complaint

    def test_the_captions_are_shown_under_the_numbers_they_belong_to(self):
        shown = panel._described(["a balcony", "a red coat"])

        assert "Image 1: a balcony" in shown
        assert shown.index("Image 1:") < shown.index("Image 2:")

    def test_the_page_says_what_the_numbers_mean(self):
        """The whole feature rests on the user and the model agreeing which
        picture is which, and an agreement one side was never told is not one."""
        assert "Image 1" in panel.NUMBERING_NOTE
        assert "not Krea syntax" in panel.NUMBERING_NOTE


class TestVisionIsRequiredOnlyForReferences:
    class Blind:
        sees = False

    class Seeing:
        sees = True

    def test_a_text_only_request_never_asks_whether_the_model_can_see(self, store, host,
                                                                      monkeypatch):
        asked = []
        monkeypatch.setattr(panel.mc_llm_runtime, "config",
                            lambda: asked.append(True) or self.Blind())
        monkeypatch.setattr(panel.sessions, "krea",
                            lambda *a, **k: iter([sessions.Event(sessions.DONE, "a prompt")]))

        frames = list(panel._generate("a rainy street", 7, 1, None, None, None, None))

        assert asked == []
        assert frames[-1][1] == "a prompt"

    def test_references_on_a_blind_model_are_refused_before_generation(self, store, host,
                                                                       monkeypatch):
        """§8: fail before generation, explain, and never quietly drop the
        pictures and write a text-only prompt instead."""
        started = []
        monkeypatch.setattr(panel.mc_llm_runtime, "config", lambda: self.Blind())
        monkeypatch.setattr(panel.sessions, "krea", lambda *a, **k: started.append(True))

        frames = list(panel._generate("edit it", 7, 1, "/tmp/a.png", None, None, None))

        assert started == []
        assert len(frames) == 1
        assert "no vision projector" in frames[0][3]

    def test_a_gap_is_refused_before_the_runtime_is_consulted(self, store, host,
                                                              monkeypatch):
        monkeypatch.setattr(panel.mc_llm_runtime, "config", lambda: self.Seeing())
        started = []
        monkeypatch.setattr(panel.sessions, "krea", lambda *a, **k: started.append(True))

        frames = list(panel._generate("edit it", 7, 1, "/tmp/a.png", None, "/tmp/c.png", None))

        assert started == []
        assert "Image 2 is empty" in frames[0][3]

    def test_the_order_shown_in_the_ui_is_the_order_handed_to_the_session(
            self, store, host, monkeypatch):
        handed = {}
        monkeypatch.setattr(panel.mc_llm_runtime, "config", lambda: self.Seeing())
        monkeypatch.setattr(panel.ui, "data_url", lambda path: f"data:{path}")

        def record(prompt, references, seed, cancel, creativity=None):
            handed["references"] = list(references)
            handed["creativity"] = creativity
            return iter([sessions.Event(sessions.DONE, "a prompt")])

        monkeypatch.setattr(panel.sessions, "krea", record)

        list(panel._generate("use image 1 and image 2", 7, 1,
                             "/tmp/first.png", "/tmp/second.png", None, None))

        assert [(r.ui_index, r.path) for r in handed["references"]] == [
            (1, "/tmp/first.png"), (2, "/tmp/second.png")]
        assert handed["references"][0].data_url == "data:/tmp/first.png"

    def test_a_picture_that_cannot_be_read_is_reported_against_its_slot(
            self, store, host, monkeypatch):
        monkeypatch.setattr(panel.mc_llm_runtime, "config", lambda: self.Seeing())
        monkeypatch.setattr(panel.ui, "data_url",
                            lambda path: None if "second" in path else f"data:{path}")
        started = []
        monkeypatch.setattr(panel.sessions, "krea", lambda *a, **k: started.append(True))

        frames = list(panel._generate("use them", 7, 1,
                                      "/tmp/first.png", "/tmp/second.png", None, None))

        assert started == []
        assert "Image 2 could not be read" in frames[0][3]


class TestTheCaptionsArriveInOrder:
    def test_the_nth_caption_event_becomes_image_n(self, store, host, monkeypatch):
        """§10: version 1 does not extend Event with image metadata, because
        caption events arrive in deterministic order. This is that guarantee
        being relied on, and therefore being tested."""
        monkeypatch.setattr(panel.mc_llm_runtime, "config",
                            lambda: TestVisionIsRequiredOnlyForReferences.Seeing())
        monkeypatch.setattr(panel.ui, "data_url", lambda path: f"data:{path}")
        monkeypatch.setattr(panel.sessions, "krea", lambda *a, **k: iter([
            sessions.Event(sessions.CAPTION, "a balcony"),
            sessions.Event(sessions.CAPTION, "a red coat"),
            sessions.Event(sessions.DONE, "the prompt")]))

        frames = list(panel._generate("use them", 7, 1, "/tmp/a.png", "/tmp/b.png", None, None))

        shown = [frame[2].get("value") for frame in frames if isinstance(frame[2], dict)
                 and frame[2].get("value")]
        assert shown[-1] == "Image 1: a balcony\n\nImage 2: a red coat"


# --------------------------------------------------------------------------- #
# History (design intent §11)
# --------------------------------------------------------------------------- #


class TestKreaHistory:
    def test_it_is_a_file_of_its_own(self, store):
        state.save_krea_session(state.KreaSession(prompt="a prompt", result="R"))
        state.save_minimax_session(state.MinimaxSession(prompt="another", result="R"))

        assert (store / "data" / state.KREA_HISTORY_FILE).exists()
        assert len(state.krea_sessions()) == 1
        assert len(state.minimax_sessions()) == 1
        assert state.prompt_sessions() == []

    def test_clearing_another_history_leaves_krea_alone(self, store):
        state.save_krea_session(state.KreaSession(prompt="a prompt"))
        state.save_minimax_session(state.MinimaxSession(prompt="another"))

        state.clear_history(state.MINIMAX_HISTORY_FILE)

        assert len(state.krea_sessions()) == 1
        assert state.minimax_sessions() == []

    def test_a_text_only_session_round_trips(self, store):
        state.save_krea_session(state.KreaSession(
            prompt="a rainy street", result="A quiet street.", seed=42))

        restored = state.krea_sessions()[0]
        assert restored.prompt == "a rainy street"
        assert restored.result == "A quiet street."
        assert restored.seed == 42
        assert restored.reference_names == [] and restored.reference_captions == []

    def test_the_reference_names_and_captions_round_trip_in_order(self, store):
        state.save_krea_session(state.KreaSession(
            prompt="replace the face in image 1", result="R",
            reference_names=["portrait.png", "identity.png"],
            reference_captions=["a woman on a balcony", "a woman in a red coat"]))

        restored = state.krea_sessions()[0]
        assert restored.reference_names == ["portrait.png", "identity.png"]
        assert restored.reference_captions == ["a woman on a balcony", "a woman in a red coat"]

    def test_no_image_data_is_ever_written(self, store):
        """§11 and §14: no image bytes, no data URLs, no temporary upload
        paths. A history file that grew a base64 JPEG per entry is one nobody
        could open."""
        panel._remember("edit it", "A quiet street.", 7,
                        [Reference(ui_index=1, path="/tmp/gradio/xyz/portrait.png",
                                   data_url="data:image/png;base64,SECRETPIXELS")],
                        ["a woman on a balcony"], 1)

        written = (store / "data" / state.KREA_HISTORY_FILE).read_text(encoding="utf-8")
        assert "SECRETPIXELS" not in written
        assert "base64" not in written
        assert "/tmp/gradio" not in written
        assert "portrait.png" in written

    def test_unknown_fields_from_a_newer_version_are_ignored_not_fatal(self, store):
        import json

        path = store / "data" / state.KREA_HISTORY_FILE
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"version": 99, "sessions": [
            {"identifier": "abc", "prompt": "a prompt", "reference_roles": ["identity"]}]}),
            encoding="utf-8")

        restored = state.krea_sessions()

        assert len(restored) == 1 and restored[0].prompt == "a prompt"

    def test_deleting_removes_only_the_named_session(self, store):
        kept = state.save_krea_session(state.KreaSession(prompt="kept"))
        state.save_krea_session(state.KreaSession(prompt="dropped", identifier="drop-me"))

        state.delete_krea_session("drop-me")

        assert [s.identifier for s in state.krea_sessions()] == [kept.identifier]

    def test_a_loaded_session_does_not_pretend_the_pictures_are_still_attached(self, store):
        """§11: restore the text, the prompt, the captions and the names as
        information -- re-generating means re-uploading."""
        saved = state.save_krea_session(state.KreaSession(
            prompt="replace the face in image 1", result="A quiet street.", seed=9,
            reference_names=["portrait.png"], reference_captions=["a woman on a balcony"]))

        prompt, written, captions, status = panel._load_session(saved.identifier)

        assert prompt == "replace the face in image 1"
        assert written == "A quiet street."
        assert "Image 1: a woman on a balcony" in captions.get("value")
        assert "portrait.png" in status
        assert "not saved with a session" in status

    def test_the_history_label_says_how_many_references_there_were(self, store):
        session = state.KreaSession(prompt="edit it", reference_names=["a.png", "b.png"])

        assert "2 refs" in session.label


class TestAPictureThatWillNotEncode:
    def test_the_slot_is_named_and_the_path_is_not(self, store, host, monkeypatch):
        """§14: a temporary upload path is somebody's home directory and often
        the name of the project they are working on. The slot number is the
        thing a user can act on anyway."""
        monkeypatch.setattr(panel.mc_llm_runtime, "config",
                            lambda: TestVisionIsRequiredOnlyForReferences.Seeing())

        def refuse(path):
            if "second" in path:
                raise OSError("cannot identify image file")
            return f"data:{path}"

        monkeypatch.setattr(panel.ui, "data_url", refuse)
        started = []
        monkeypatch.setattr(panel.sessions, "krea", lambda *a, **k: started.append(True))

        frames = list(panel._generate("use them", 7, 1,
                                      "/tmp/gradio/first.png", "/tmp/gradio/second.png",
                                      None, None))

        assert started == []
        assert "Image 2 could not be read" in frames[0][3]
        assert "/tmp/gradio" not in frames[0][3]
