"""Counting a conversation in the model's own tokens (``mc_llm_tokens``).

The old trimmer estimated 3.2 characters a token; the user's Gemma 4 thread runs
at about 4.1, so a fifth of the room went unused. A running llama-server is
asked instead -- ``/tokenize`` for a message, ``/apply-template`` then
``/tokenize`` for a whole request -- and without one the thread's own measured
ratio is used. Nothing here starts a server.
"""

from __future__ import annotations

import pytest

import mc_llm_tokens as tokens


class Server:
    """llama-server's two routes, counting a token per word and saying what
    it was asked."""

    base_url, api_key = "http://127.0.0.1:1", "k"

    def __init__(self, fail_after=None):
        self.asked = []
        self.fail_after = fail_after

    def post(self, route, body):
        self.asked.append((route, body))
        if self.fail_after is not None and len(self.asked) > self.fail_after:
            raise ConnectionError("gone")
        if route == "/apply-template":
            rendered = "".join(f"<{message['role']}>{message['content']}<end>"
                               for message in body["messages"])
            return {"prompt": rendered}
        words = body["content"].split()
        return {"tokens": list(range(len(words) + (1 if body.get("add_special") else 0)))}


@pytest.fixture(autouse=True)
def fresh():
    tokens.forget()
    yield
    tokens.forget()


def counter_for(server, ratio=0.0):
    counter = tokens.Counter(server, model="m", ratio=ratio)
    counter._post = server.post
    return counter


class TestWithAServer:
    def test_a_text_is_counted_by_the_server(self):
        server = Server()

        assert counter_for(server).text("one two three") == 3
        assert server.asked[-1] == ("/tokenize", {"content": "one two three",
                                                  "add_special": False,
                                                  "parse_special": True})

    def test_a_text_is_counted_once(self):
        server = Server()
        counter = counter_for(server)

        counter.text("one two three")
        counter.text("one two three")
        counter_for(server).text("one two three")

        assert len(server.asked) == 1

    def test_never_for_another_model(self):
        server = Server()
        counter_for(server).text("one two three")

        other = tokens.Counter(server, model="another")
        other._post = server.post
        other.text("one two three")

        assert len(server.asked) == 2

    def test_a_message_costs_its_text_its_markers_and_its_picture(self):
        counter = counter_for(Server())

        assert counter.message("one two") == 2 + tokens.MESSAGE_OVERHEAD
        assert counter.message("one two", still=True) == 2 + tokens.MESSAGE_OVERHEAD + 300

    def test_a_request_is_rendered_by_the_template_then_counted_with_its_specials(self):
        server = Server()
        wire = [{"role": "system", "content": "be kind"},
                {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "x"}},
                                             {"type": "text", "text": "look at this"}]}]

        counted, exact = counter_for(server).prompt(wire)

        assert exact is True
        route, body = server.asked[0]
        assert route == "/apply-template"
        assert body["messages"][1] == {"role": "user", "content": "look at this"}
        assert server.asked[1][1]["add_special"] is True
        # The rendered words, the special token added, and the picture as a still.
        assert counted == len("<system>be kind<end><user>look at this<end>".split()) + 1 + 300


class TestWithoutOne:
    def test_estimated_at_the_builders_ratio(self):
        counter = tokens.Counter(None)

        assert counter.exact is False
        assert counter.text("x" * 32) == 10

    def test_estimated_at_the_threads_own_measured_ratio(self):
        assert tokens.Counter(None, ratio=0.25).text("x" * 32) == 8

    def test_a_request_is_the_sum_of_its_messages_and_says_so(self):
        counted, exact = tokens.Counter(None, ratio=0.25).prompt(
            [{"role": "system", "content": "x" * 40}, {"role": "user", "content": "y" * 8}])

        assert exact is False
        assert counted == (10 + tokens.MESSAGE_OVERHEAD) + (2 + tokens.MESSAGE_OVERHEAD)

    def test_a_server_that_stops_answering_is_estimated_from_then_on(self):
        server = Server(fail_after=1)
        counter = counter_for(server, ratio=0.25)

        assert counter.text("one two three") == 3
        assert counter.text("x" * 32) == 8
        assert counter.exact is False
        asked = len(server.asked)
        counter.text("y" * 40)
        assert len(server.asked) == asked, "nothing more is asked of a server that went"


class TestForTheRunningServer:
    def test_none_running_is_an_estimate_and_starts_nothing(self, monkeypatch):
        import mc_llm_sessions as sessions

        class Runtime:
            asked = []

            def warm_client(self, needs_vision=False):
                self.asked.append(needs_vision)
                return None

        runtime = Runtime()
        monkeypatch.setattr(sessions, "_runtime_for",
                            lambda role="": sessions._Resolved(runtime, None, None))

        counter = tokens.for_server(ratio=0.3)

        assert counter.exact is False and counter.client is None
        assert runtime.asked == [False]


class PieceServer:
    """``/tokenize`` with pieces, for a tokenizer that puts a space before the
    start of a text, as SentencePiece's own does: ``" rain"`` alone reads as a
    lone space and the word, after another token as the word."""

    base_url, api_key = "http://127.0.0.1:1", "k"

    def __init__(self, prefix_space=True):
        self.asked = []
        self.prefix_space = prefix_space

    def post(self, route, body):
        self.asked.append((route, body))
        assert route == "/tokenize" and body["with_pieces"] is True
        assert body["add_special"] is False and body["parse_special"] is False
        text = body["content"]
        if self.prefix_space:
            text = " " + text
        found = []
        for index, word in enumerate(text.replace("*", " * ").split(" ")):
            if word == "" and index == 0 and self.prefix_space:
                found.append({"id": 1, "piece": " "})
            elif word:
                found.append({"id": 100 + sum(map(ord, word)), "piece": " " + word})
        return {"tokens": found}


class TestPiecesOfAWord:
    def counter(self, server):
        counter = tokens.Counter(server, model="m")
        counter._post = server.post
        return counter

    def test_a_word_is_read_after_another_token_so_a_leading_space_is_not_a_piece(self):
        server = PieceServer(prefix_space=True)

        found = self.counter(server).pieces(" rain")

        assert [piece for _, piece in found] == [" rain"]
        assert [body["content"] for _, body in server.asked] == [
            tokens.PIECE_CONTEXT + " rain", tokens.PIECE_CONTEXT]

    def test_pieces_are_remembered_per_server_and_model(self):
        server = PieceServer()
        counter = self.counter(server)

        counter.pieces(" rain")
        counter.pieces(" rain")
        assert len(server.asked) == 2

        other = tokens.Counter(server, model="another")
        other._post = server.post
        other.pieces(" rain")
        assert len(server.asked) == 4

    def test_when_the_context_does_not_stay_whole_the_word_is_read_alone(self):
        class Merging(PieceServer):
            def post(self, route, body):
                self.asked.append((route, body))
                if body["content"] == tokens.PIECE_CONTEXT:
                    return {"tokens": [{"id": 9, "piece": "*"}]}
                if body["content"].startswith(tokens.PIECE_CONTEXT):
                    return {"tokens": [{"id": 77, "piece": "*rain"}]}
                return {"tokens": [{"id": 5, "piece": "rain"}]}

        found = self.counter(Merging()).pieces("rain")

        assert found == [(5, "rain")]

    def test_a_piece_that_is_not_text_is_read_from_its_bytes(self):
        class Bytes(PieceServer):
            def post(self, route, body):
                if body["content"] == tokens.PIECE_CONTEXT:
                    return {"tokens": [{"id": 9, "piece": "*"}]}
                return {"tokens": [{"id": 9, "piece": "*"},
                                   {"id": 3, "piece": [0xC3, 0xA9]}]}

        assert self.counter(Bytes()).pieces("é") == [(3, "é")]

    def test_with_no_server_or_a_server_that_fails_there_are_no_pieces(self):
        assert tokens.Counter(None).pieces(" rain") is None

        class Failing(PieceServer):
            def post(self, route, body):
                raise ConnectionError("gone")

        counter = self.counter(Failing())
        assert counter.pieces(" rain") is None
        assert counter.exact is False

    def test_for_client_names_the_model_the_way_for_server_does(self, monkeypatch):
        import mc_llm_runtime

        monkeypatch.setattr(mc_llm_runtime, "config", lambda role="": type(
            "Config", (), {"model": "/models/gemma.gguf"})())
        server = PieceServer()

        made = tokens.for_client(server)

        assert made.exact is True and made.server.startswith(server.base_url + "|")
        assert tokens.for_client(None).exact is False
