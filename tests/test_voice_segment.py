"""Where a reply may be cut, and every way that decision goes wrong.

Streaming speech is only as good as its boundaries. A segmenter that commits
too early says each word with the intonation of a full stop; one that commits
too late is a feature nobody can hear the point of; and one that commits in the
wrong *place* says "the value is three" and then, a second later, "point one
four one five nine" as though a new sentence had begun.

Every test here is one of those, and section 8's guard list is the index:
decimals, abbreviations, domains, ellipses, fences, and the label
``clean_reply`` removes after the fact. The tests are about text only -- no
audio, no worker, no queue -- because :mod:`mc_voice_segment` is deliberately
the one part of this feature that has no I/O in it at all.
"""

from __future__ import annotations

import mc_voice_segment as segment


def run(chunks, labels=("Alice", "You", "Assistant"), final=None):
    """Feed ``chunks`` one at a time and flush. Returns every segment."""
    machine = segment.Segmenter(labels=labels)
    found = []
    for chunk in chunks:
        found += machine.feed(chunk)
    found += machine.flush(final)
    return found


class TestWhatCommitsAndWhen:
    def test_t_seg_1_a_short_complete_first_sentence_commits(self):
        """Section 8 allows it explicitly, and it is the whole difference
        between "Yes, that's possible." arriving now and arriving after the
        paragraph that follows it."""
        machine = segment.Segmenter()
        assert machine.feed("Yes, that's possible.") == ["Yes, that's possible."]

    def test_t_seg_2_no_segment_is_a_raw_token(self):
        """Tokens arrive two or three characters at a time. Synthesising one is
        how a reply comes out sounding like a list of words."""
        machine = segment.Segmenter()
        text = ("The quick brown fox jumps over the lazy dog and then it keeps going for a "
                "while longer so that there is enough text here to commit something.")
        found = []
        for index in range(0, len(text), 3):
            found += machine.feed(text[index:index + 3])
        assert all(len(item) >= segment.SHORT_SENTENCE for item in found), found

    def test_t_seg_3_a_paragraph_beats_a_weaker_boundary(self):
        found = run(["First thought, which is a whole one.\n\nSecond thought follows it."])
        assert found[0] == "First thought, which is a whole one."

    def test_nothing_commits_before_there_is_a_boundary(self):
        machine = segment.Segmenter()
        assert machine.feed("This sentence has not finished yet and there is no") == []
        assert machine.pending

    def test_a_short_first_sentence_commits_even_with_the_next_one_behind_it(self):
        """The streaming-edge condition, and the reason it is not cosmetic.

        A model that writes "Yes, that's possible. Here is the" in one delta has
        finished the first sentence at exactly the moment one that wrote it
        alone did. Holding it back because something arrived behind it made the
        first segment's latency a property of chunk boundaries.
        """
        machine = segment.Segmenter()
        found = machine.feed("Yes, that's possible. Here is the next sentence beginning")
        assert found == ["Yes, that's possible."]
        assert machine.pending == " Here is the next sentence beginning"

    def test_the_opening_is_still_a_sentence_and_not_a_fragment(self):
        """SHORT_SENTENCE is what stops the correction above from turning a
        numbered list into speech one item at a time: the full stop after "1."
        is a real boundary by every guard this module has, and two characters
        is not a sentence."""
        machine = segment.Segmenter()
        found = machine.feed("1. First item here, and then rather more of it follows.")
        assert found == ["First item here, and then rather more of it follows."]

    def test_no_text_is_lost_or_repeated_across_the_correction(self):
        machine = segment.Segmenter()
        found = machine.feed("Yes, that's possible. Here is the next sentence beginning")
        found += machine.feed(" and here is the end of it.")
        found += machine.flush()
        joined = " ".join(found)
        assert joined.count("Yes, that's possible.") == 1
        assert "Here is the next sentence beginning and here is the end of it." in joined

    def test_the_second_segment_does_not_wait_for_the_ordinary_target(self):
        """The first-to-second gap, stated as the thing that causes it: sentence
        one is playing, and sentence two used to need a hundred characters."""
        machine = segment.Segmenter()
        assert machine.feed("Yes, that's possible.") == ["Yes, that's possible."]
        text = "It takes a little more than fifty characters to say this."
        assert len(text) < segment.TARGET, "this test would pass for the wrong reason"
        assert machine.feed(" " + text) == [text]

    def test_later_segments_keep_the_ordinary_target(self):
        """Only the second one is hurried. By the third there is audio queued in
        front of it, and a longer segment sounds better for free."""
        machine = segment.Segmenter()
        machine.feed("Yes, that's possible.")
        machine.feed(" It takes a little more than fifty characters to say this.")
        assert machine.segments == 2
        assert machine.feed(" A third sentence of about sixty characters goes here.") == []
        assert machine.pending.strip().startswith("A third")

    def test_the_targets_are_ordered_the_way_the_latency_is(self):
        machine = segment.Segmenter()
        assert machine.first_target >= machine.second_target
        assert machine.second_target <= machine.target


class TestTheOpeningsEnvelope:
    """Time to the first spoken word is, almost exactly, the length of the first
    sentence.

    Per-unit timings from an ordinary session put Kokoro at ~28 ms of synthesis
    per character with ~110 ms of fixed cost per call, and the first callback
    block is one whole sentence -- so a 28-character opening was speaking in a
    second and a 95-character one took nearly three. Nothing else in the chain
    is anywhere near that big: the browser's prebuffer contributed 0-8 ms on
    every measured turn.
    """

    def first_of(self, text):
        machine = segment.Segmenter()
        found = machine.feed(text)
        return found[0] if found else None

    def test_a_short_opening_sentence_is_untouched(self):
        assert self.first_of("Yes, that's possible. And a good deal more after it.") == (
            "Yes, that's possible.")

    def test_a_long_opening_sentence_is_cut_at_a_comma(self):
        found = self.first_of(
            "It turns out that the answer is a little involved, and it depends on a "
            "few things. Then a second sentence.")
        assert found == "It turns out that the answer is a little involved,"
        assert len(found) <= segment.FIRST_SOFT_MAX

    def test_a_long_opening_sentence_with_no_early_break_is_spoken_whole(self):
        """The trade, stated as a test. A comma is a pause Kokoro would have
        taken anyway; a word boundary in the middle of the first thing anybody
        hears is a seam, and that is not where to spend one."""
        found = self.first_of(
            "It turns out that the answer here is really quite a lot more involved "
            "than that. Then a second sentence.")
        assert found.endswith("than that."), found
        assert len(found) > segment.FIRST_SOFT_MAX

    def test_the_opening_takes_the_first_sentence_and_not_everything_that_fits(self):
        """Rule 3 takes the last boundary that fits, which is right for an
        ordinary segment and wrong here: every character before the full stop is
        time nobody is hearing anything."""
        found = self.first_of(
            "It turns out that the answer here is really quite a lot more involved "
            "than that. Then a second one. And a third.")
        assert found.count(".") == 1, found

    def test_a_punctuation_free_opening_stops_at_the_ceiling(self):
        found = self.first_of(" ".join(["alpha"] * 60))
        assert len(found) <= segment.FIRST_HARD_MAX
        assert found.endswith("alpha"), "the cut landed inside a word"

    def test_the_ceiling_is_far_below_the_ordinary_one(self):
        machine = segment.Segmenter()
        assert machine.first_soft_max < machine.second_soft_max < machine.soft_max
        assert machine.first_hard_max < machine.second_hard_max < machine.hard_max
        assert machine.first_target <= machine.first_soft_max

    def test_the_guards_are_not_relaxed_by_the_smaller_envelope(self):
        """A decimal, an abbreviation, an initial and a domain are not
        boundaries at seventy characters either."""
        found = self.first_of(
            "Dr. Smith measured 3.14159 at example.com/page and then wrote it all up "
            "in a good deal more detail than anybody wanted.")
        joined = found or ""
        for hazard in ("Dr.", "3.14159", "example.com/page"):
            assert not joined.rstrip().endswith(hazard.rstrip(".")), joined

    def test_nothing_is_lost_across_the_cut(self):
        text = ("It turns out that the answer is a little involved, and it depends on a "
                "few things. Then a second sentence.")
        machine = segment.Segmenter()
        found = machine.feed(text) + machine.flush()
        rebuilt = " ".join(found)
        for word in ("involved", "depends", "second", "sentence"):
            assert rebuilt.count(word) == 1, rebuilt


class TestTheSecondUnitsEnvelope:
    """How soon the second segment commits is one question; how big it is
    allowed to get is another, and only the first had an answer.

    A model that writes a hundred and eighty characters before its first full
    stop hands the second segment one enormous sentence. It arrives as text
    quickly -- SECOND_TARGET is satisfied immediately -- and then takes as long
    to synthesise as everything it was supposed to be covering. Continuity is
    lost to the size of the unit rather than to the wait for it.
    """

    RUN = ("It turns out that the answer here involves a fairly long explanation "
           "which runs on for quite a while without ever reaching a full stop and "
           "keeps going, adding clause after clause, well past two hundred "
           "characters before it finally ends.")

    def second_of(self, machine, text):
        assert machine.feed("Yes, that's possible.") == ["Yes, that's possible."]
        return machine.feed(" " + text)

    def test_a_long_second_unit_is_cut_inside_the_envelope(self):
        machine = segment.Segmenter()
        found = self.second_of(machine, self.RUN)
        assert found, "the second unit was never committed"
        assert len(found[0]) <= segment.SECOND_HARD_MAX, len(found[0])
        assert len(found[0]) > segment.SECOND_TARGET, "it was cut smaller than the target"

    def test_a_clause_inside_the_soft_limit_is_preferred_to_the_ceiling(self):
        """The envelope is where a weaker boundary becomes acceptable, not a
        character count to chop at. A comma within reach is taken; the word
        boundary at the ceiling is what happens when there is nothing better."""
        text = ("The explanation here is a little involved and it carries on, "
                + "and onwards " * 20)
        assert text.index(",") < segment.SECOND_SOFT_MAX
        assert len(text) > segment.SECOND_SOFT_MAX
        assert "." not in text, "a sentence boundary would beat the clause"
        machine = segment.Segmenter()
        found = self.second_of(machine, text)
        assert found, "nothing committed"
        assert found[0].rstrip().endswith(","), found[0][-40:]
        assert len(found[0]) <= segment.SECOND_SOFT_MAX

    def test_a_punctuation_free_second_unit_ends_at_a_word(self):
        words = " ".join(["alpha", "bravo", "charlie", "delta", "echo"] * 12)
        machine = segment.Segmenter()
        found = self.second_of(machine, words)
        assert found, "nothing committed"
        assert len(found[0]) <= segment.SECOND_HARD_MAX
        assert words.startswith(found[0])
        assert words[len(found[0])] == " ", "the cut landed inside a word"

    def test_the_final_tail_cannot_escape_the_envelope(self):
        """T-7. Generation ending before another segment arrived is not a reason
        for the second unit to become the whole rest of the reply."""
        machine = segment.Segmenter()
        assert machine.feed("Yes, that's possible.") == ["Yes, that's possible."]
        found = machine.flush("Yes, that's possible. " + self.RUN)
        assert len(found) >= 2, [len(item) for item in found]
        assert len(found[0]) <= segment.SECOND_HARD_MAX, len(found[0])

    def test_later_units_keep_the_ordinary_envelope(self):
        """Only the second one is small. By the third there is audio queued in
        front of it, and a longer segment sounds better for free."""
        machine = segment.Segmenter()
        machine.feed("Yes, that's possible.")
        machine.feed(" And here is a second complete sentence, long enough to commit.")
        assert machine.segments == 2
        found = machine.feed(" " + self.RUN)
        assert found, "the third unit never committed"
        assert len(found[0]) > segment.SECOND_HARD_MAX, (
            "the third unit was held to the second unit's envelope")

    def test_the_envelope_still_refuses_every_unsafe_boundary(self):
        """The limits move; the guards do not. A decimal, an abbreviation, a
        domain and an initial are not boundaries at 220 characters either."""
        filler = "the value is measured again and again and again, "
        text = (filler * 4) + "Dr. Smith wrote 3.14159 to example.com about it, J. R. R. Tolkien too."
        machine = segment.Segmenter()
        found = self.second_of(machine, text)
        joined = " ".join(found)
        for hazard in ("Dr.", "3.14159", "example.com", "J. R. R."):
            if hazard in joined:
                assert not joined.rstrip().endswith(hazard.rstrip(".")), joined[-40:]
        assert "3.14" not in machine.pending or "3.14159" in machine.pending

    def test_a_fence_is_not_split_by_the_smaller_ceiling(self):
        machine = segment.Segmenter()
        machine.feed("Yes, that's possible.")
        fenced = "```\n" + "\n".join(f"line_{index} = {index}" for index in range(40)) + "\n"
        assert len(fenced) > segment.SECOND_HARD_MAX
        assert machine.feed(" " + fenced) == []

    def test_the_limits_are_ordered(self):
        machine = segment.Segmenter()
        assert machine.second_target <= machine.second_soft_max <= machine.second_hard_max
        assert machine.second_hard_max < machine.hard_max
        assert machine.second_soft_max < machine.soft_max


class TestTheGuards:
    def test_t_seg_4_a_decimal_point_is_not_a_sentence(self):
        found = run(["The value of pi is 3.14159 and it goes on for ever, which is a fact "
                     "that many people find quite remarkable."])
        assert not any(item.endswith("3.") for item in found), found
        assert "3.14159" in " ".join(found)

    def test_t_seg_5_a_common_abbreviation_is_not_a_sentence(self):
        found = run(["Dr. Smith went to St. Mary's on Tuesday with Mr. Jones and they were "
                     "there for most of the afternoon."])
        assert len(found) == 1, found

    def test_an_initial_is_not_a_sentence(self):
        """Asserted as "no cut after an initial" rather than "one segment": the
        opening envelope may take the comma after Tolkien, which is a clause
        boundary and a pause a reader would take anyway. What must never happen
        is a cut after ``J.``, which would say "jay" and then start again."""
        found = run(["The author is J. R. R. Tolkien, who wrote the book that everybody has "
                     "heard of and quite a few other ones too."])
        assert "J. R. R. Tolkien" in " ".join(found), found
        for item in found:
            assert not item.rstrip().endswith(("J.", "R.")), found

    def test_t_seg_6_a_domain_dot_is_not_a_sentence(self):
        found = run(["Have a look at example.com/page for the details, which are all written "
                     "out there in a good deal more depth than here."])
        assert "example.com/page" in " ".join(found)
        assert not any(item.endswith("example.") for item in found), found

    def test_t_seg_7_a_markdown_fence_is_not_split_inside(self):
        found = run(["Here is some code:\n\n```python\nx = 1.0\ny = 2.0\n```\n\nAnd that is "
                     "what it does, which should be reasonably clear by now."])
        joined = " ".join(found)
        assert "x = 1.0" in joined and "y = 2.0" in joined
        assert "```" not in joined, "a fence marker was going to be read aloud"

    def test_an_ellipsis_is_one_pause_and_not_three(self):
        """Same shape: the comma may be taken, the dots may not."""
        found = run(["Well... I am really not certain about that at all, if I am honest with "
                     "you about the whole business."])
        assert "Well... I am really not certain" in " ".join(found), found
        for item in found:
            assert not item.rstrip().endswith(("Well.", "Well..")), found

    def test_t_seg_8_punctuation_free_text_splits_only_at_a_word_boundary(self):
        found = run([" ".join(["alpha"] * 400)])
        assert len(found) > 1
        for item in found:
            assert not item.startswith("lpha") and not item.endswith("alph"), item
            assert len(item) <= segment.HARD_MAX

    def test_a_single_enormous_word_still_ends_at_the_ceiling(self):
        """The one case where cutting inside a word is the only option, and the
        one where refusing to cut would mean never speaking at all."""
        found = run(["x" * 2000])
        assert found and max(len(item) for item in found) <= segment.HARD_MAX


class TestTheLeadingLabel:
    def test_t_seg_9_a_label_clean_reply_removes_is_never_spoken(self):
        found = run(["Alice: ", "Hello there, this is the first thing said in the reply."])
        assert not any(item.lower().startswith("alice:") for item in found), found
        assert found[0].startswith("Hello there")

    def test_the_opening_is_held_only_until_the_question_is_settled(self):
        """It costs the first segment nothing in practice: deciding takes at
        most one short line, and the first-segment target is longer."""
        machine = segment.Segmenter(labels=("Alice",))
        assert machine.feed("Al") == []
        assert machine.feed("ice: Hello there, and here is a whole first sentence for it.")

    def test_a_colon_that_is_not_a_label_is_left_alone(self):
        found = run(["Here is the thing: it works, which is the answer to the question you "
                     "actually asked me about."])
        assert "Here is the thing" in found[0]
        assert "it works" in " ".join(found)

    def test_a_reply_with_no_label_is_not_held_back_for_ever(self):
        machine = segment.Segmenter(labels=("Alice",))
        assert machine.feed("Hello there, this is a first sentence with no label at all.")


class TestTheAuthoritativeFinalText:
    def test_t_seg_10_the_final_text_is_spoken_exactly_once(self):
        machine = segment.Segmenter(labels=("Assistant",))
        spoken = machine.feed("Hello there, this is a first sentence for the test. ")
        spoken += machine.flush("Hello there, this is a first sentence for the test. "
                                "And here is the tail of it.")
        joined = " ".join(spoken)
        assert joined.count("this is a first sentence") == 1, joined
        assert "And here is the tail of it." in joined

    def test_a_label_the_panel_strips_does_not_re_speak_the_reply(self):
        """``clean_reply`` removes the label from the *final* text, so the final
        text and the streamed text differ at character zero. Aligning them by
        bytes would replay the whole reply."""
        machine = segment.Segmenter(labels=("Assistant",))
        spoken = machine.feed("Assistant:  Hello there, this is a fairly long first sentence. ")
        spoken += machine.flush("Hello there, this is a fairly long first sentence. Tail.")
        joined = " ".join(spoken)
        assert joined.count("Hello there") == 1, joined
        assert joined.endswith("Tail.")

    def test_t_seg_11_a_continuation_speaks_only_the_new_tail(self):
        """Section 7. The existing opening is on screen already and may have been
        read aloud once; re-speaking it is the surprising behaviour."""
        opening = "This is the opening that already existed. "
        machine = segment.Segmenter()
        spoken = machine.feed("And this is the newly generated tail of the reply.")
        spoken += machine.flush()
        assert opening.strip() not in " ".join(spoken)
        assert "newly generated tail" in " ".join(spoken)

    def test_the_tail_is_bounded_like_every_other_segment(self):
        machine = segment.Segmenter()
        found = machine.flush(" ".join(["alpha"] * 400))
        assert all(len(item) <= segment.HARD_MAX for item in found)


class TestNormalization:
    def test_emphasis_marks_are_removed_and_words_are_not(self):
        assert segment.speech_text("This is **important** and _urgent_.") == \
            "This is important and urgent."

    def test_a_heading_loses_its_hashes_and_keeps_its_words(self):
        assert segment.speech_text("## The Plan") == "The Plan"

    def test_a_link_keeps_its_label_and_loses_its_address(self):
        assert segment.speech_text("See [the docs](https://example.com/a) for more.") == \
            "See the docs for more."

    def test_a_bare_url_in_running_text_is_left_alone(self):
        """Section 10: do not silently discard URLs. Special handling for them
        is future scope, and future scope is not "quietly drop it"."""
        assert "https://example.com/a" in segment.speech_text("Go to https://example.com/a now.")

    def test_code_and_numbers_survive(self):
        found = segment.speech_text("Run `pip install x` at 3.14 per cent.")
        assert "pip install x" in found and "3.14" in found

    def test_a_list_becomes_pauses_and_not_bullet_characters(self):
        found = segment.speech_text("- first\n- second")
        assert "-" not in found
        assert "first" in found and "second" in found
