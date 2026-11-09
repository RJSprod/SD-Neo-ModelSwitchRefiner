"""The language model's control vector (``mc_llm_control_vector``).

The third way of injecting words into the model's mind (2026-11-09): a direction
per layer, added to the model's hidden state, given to llama-server when it
starts. These hold the file as llama.cpp reads one (``direction.N`` rows of
32-bit numbers, layer 0 never), the check against the model it runs -- a vector
is one model's, by its width and its layers -- the choice kept per model, the
copy a start is given with the strength and layers taken in (because
``--control-vector-scaled`` splits ``FILE:STRENGTH`` at the colon a Windows path
already has), the start that sets a vector aside when llama-server will not
load it, the saved prompt caches kept apart, and the model stopped to take a new
choice only when nothing is being written. And that no file name reaches a log.
"""

from __future__ import annotations

import logging
import struct
import threading
import time
from pathlib import Path

import pytest

import mc_gguf
import mc_llm_control_vector as cv
import mc_llm_paths
import mc_llm_runtime
import mc_llm_slot_cache


def _kv_string(key, value):
    encoded_key, encoded = key.encode(), value.encode()
    return (struct.pack("<Q", len(encoded_key)) + encoded_key + struct.pack("<I", mc_gguf.STRING)
            + struct.pack("<Q", len(encoded)) + encoded)


def _kv_u32(key, value):
    encoded_key = key.encode()
    return (struct.pack("<Q", len(encoded_key)) + encoded_key + struct.pack("<I", mc_gguf.UINT32)
            + struct.pack("<I", value))


def model_file(folder: Path, width=8, layers=12, name="gemma-test.gguf") -> Path:
    """A model's header, as much of one as the check reads."""
    path = folder / name
    path.write_bytes(b"GGUF" + struct.pack("<I", 3) + struct.pack("<QQ", 0, 3)
                     + _kv_string("general.architecture", "gemma4")
                     + _kv_u32("gemma4.embedding_length", width)
                     + _kv_u32("gemma4.block_count", layers))
    mc_gguf.forget(path)
    return path


class Config:
    def __init__(self, model):
        self.model = model


@pytest.fixture(autouse=True)
def data(tmp_path, monkeypatch):
    root = tmp_path / "llm"
    root.mkdir()
    monkeypatch.setattr(mc_llm_paths, "data_root", lambda: root)
    monkeypatch.setattr(mc_llm_runtime, "runtime_supports", lambda flag, configuration=None:
                        flag == cv.FLAG)
    cv._given.clear()
    cv._set_aside.clear()
    cv._waiting.clear()
    yield root
    cv._given.clear()
    cv._set_aside.clear()
    cv._waiting.clear()


@pytest.fixture
def model(tmp_path, monkeypatch):
    path = model_file(tmp_path)
    configuration = Config(path)
    monkeypatch.setattr(mc_llm_runtime, "config", lambda role="": configuration)
    monkeypatch.setattr(cv, "_servers_of", lambda model: [])
    return path


def vector(name="happy.gguf", width=8, layers=range(1, 12), value=None, hint="gemma4"):
    path = cv.folder() / name
    cv.write(path, {layer: [float(layer if value is None else value)] * width
                    for layer in layers}, hint)
    return path


# -- the file ---------------------------------------------------------------------- #


class TestTheFile:
    def test_a_vector_written_is_read_back_direction_by_direction(self):
        path = vector()

        found, directions = cv.read(path)

        assert found.valid and found.width == 8 and found.layers == tuple(range(1, 12))
        assert found.hint == "gemma4"
        assert list(directions[3]) == [3.0] * 8

    def test_it_is_what_llama_cpp_writes_one_as(self):
        """``general.architecture`` controlvector, a model hint, a layer count,
        and ``direction.N`` tensors: one row of 32-bit numbers each."""
        path = vector(layers=[2, 5])

        header = mc_gguf.read(path)

        assert header.architecture == "controlvector"
        assert header.metadata["controlvector.layer_count"] == 2
        assert header.tensor_count == 2

    @pytest.mark.parametrize("damage, said", [
        (lambda path: path.write_bytes(b"NOPE" + b"\0" * 40), "not a GGUF file"),
        (lambda path: path.write_bytes(b"GGUF" + struct.pack("<I", 3) + b"\x01"),
         "could not be read"),
    ])
    def test_a_file_that_is_not_a_vector_says_why(self, damage, said):
        path = cv.folder() / "broken.gguf"
        path.parent.mkdir(parents=True, exist_ok=True)
        damage(path)

        found = cv.read(path, header_only=True)

        assert not found.valid and said in found.error

    def test_a_layer_zero_or_a_tensor_that_is_not_a_direction_is_refused(self, data):
        path = vector(layers=[1, 2])
        raw = path.read_bytes().replace(b"direction.1", b"direction.0")
        zero = cv.folder() / "zero.gguf"
        zero.write_bytes(raw)
        other = cv.folder() / "other.gguf"
        other.write_bytes(path.read_bytes().replace(b"direction.2", b"weightxyz.2"))

        assert "directions" in cv.read(zero, header_only=True).error
        assert "directions" in cv.read(other, header_only=True).error

    def test_directions_of_two_widths_are_refused(self):
        path = cv.folder() / "mixed.gguf"
        cv.write(path, {1: [1.0] * 8, 2: [1.0] * 16})

        assert "same width" in cv.read(path, header_only=True).error


class TestItFitsTheModel:
    def test_a_vector_as_wide_as_the_model_with_its_layers_fits(self, model):
        assert cv.fits(cv.read(vector(), header_only=True), model) == ""

    def test_one_made_for_another_width_does_not(self, model):
        found = cv.read(vector("wide.gguf", width=16), header_only=True)

        assert cv.fits(found, model) == "It was made for a model 16 wide; this one is 8."

    def test_one_with_layers_past_the_models_does_not(self, model):
        found = cv.read(vector("deep.gguf", layers=range(1, 30)), header_only=True)

        assert "past this model's 12 layers" in cv.fits(found, model)

    def test_the_default_layers_start_about_a_third_in_and_end_at_the_last(self):
        assert cv.default_layers(12) == (4, 11)
        assert cv.default_layers(48) == (14, 47)
        assert cv.default_layers(1) == (1, 1)

    def test_layers_chosen_are_held_in_the_models_range_and_in_order(self):
        found = cv.resolved(cv.Choice(vector="v", first_layer=40, last_layer=2), 12)

        assert (found.first_layer, found.last_layer) == (2, 11)


# -- the choice, and what a start is given ---------------------------------------- #


class TestTheChoice:
    def test_it_is_kept_for_the_model_by_its_file_name(self, model, data):
        vector()

        cv.choose("happy.gguf", strength=0.5)

        assert cv.chosen(model) == cv.Choice(vector="happy.gguf", strength=0.5)
        assert (data / cv.FILE_NAME).is_file()
        assert cv.chosen(model.with_name("other-model.gguf")) == cv.Choice()

    def test_what_is_not_given_stays_and_none_clears_the_vector(self, model):
        vector()
        cv.choose("happy.gguf", strength=0.5, first_layer=3)

        cv.choose(None, strength=2.0)
        assert cv.chosen(model) == cv.Choice(vector="happy.gguf", strength=2.0, first_layer=3)

        cv.choose("")
        assert cv.chosen(model).vector == ""

    def test_a_vector_that_does_not_fit_or_is_not_there_is_refused(self, model):
        vector("wide.gguf", width=16)

        with pytest.raises(cv.Refused, match="16 wide"):
            cv.choose("wide.gguf")
        with pytest.raises(cv.Refused, match="not in the folder"):
            cv.choose("missing.gguf")

    def test_the_strength_is_held_within_its_limit(self, model):
        vector()

        cv.choose("happy.gguf", strength=99)

        assert cv.chosen(model).strength == cv.STRENGTH_LIMIT

    def test_with_no_model_set_up_it_is_refused(self, monkeypatch):
        monkeypatch.setattr(mc_llm_runtime, "config", lambda role="": Config(None))

        with pytest.raises(cv.Refused, match="Set up a language model"):
            cv.choose("happy.gguf")


class TestWhatAStartIsGiven:
    def test_no_vector_chosen_is_no_flag(self, model):
        assert cv.launch_flags(Config(model)) == []

    def test_a_copy_with_the_strength_and_layers_taken_in_is_given(self, model):
        """Not FILE:STRENGTH: ``--control-vector-scaled`` splits at the colon, and
        ``C:\\...`` has one already."""
        vector()
        cv.choose("happy.gguf", strength=0.5, first_layer=6, last_layer=8)

        flags = cv.launch_flags(Config(model))

        assert flags[0] == cv.FLAG and len(flags) == 2
        given = Path(flags[1])
        assert given.parent == cv.folder() / cv.APPLIED_DIRNAME
        assert ":" not in given.name
        found, directions = cv.read(given)
        assert found.layers == (6, 7, 8)
        assert list(directions[6]) == [3.0] * 8

    def test_the_same_choice_is_the_same_file_and_a_new_one_replaces_it(self, model):
        vector()
        cv.choose("happy.gguf", strength=0.5)
        first = cv.launch_flags(Config(model))[1]
        assert cv.launch_flags(Config(model))[1] == first

        cv.choose(None, strength=1.5)
        second = cv.launch_flags(Config(model))[1]

        assert second != first
        assert not Path(first).exists(), "a copy no longer chosen is removed"

    def test_a_build_without_the_flag_starts_without_it(self, model, monkeypatch, caplog):
        vector()
        cv.choose("happy.gguf")
        monkeypatch.setattr(mc_llm_runtime, "runtime_supports", lambda *args: False)

        with caplog.at_level(logging.WARNING, logger="model_chain"):
            assert cv.launch_flags(Config(model)) == []
        assert "does not take a control vector" in caplog.text

    def test_a_vector_gone_from_the_folder_is_left_off_without_naming_it(self, model, caplog):
        path = vector()
        cv.choose("happy.gguf")
        path.unlink()

        with caplog.at_level(logging.WARNING, logger="model_chain"):
            assert cv.launch_flags(Config(model)) == []
        assert "left off this start" in caplog.text
        assert "happy" not in caplog.text

    def test_a_data_folder_with_a_comma_in_its_path_starts_without_one(self, model, tmp_path,
                                                                       monkeypatch):
        odd = tmp_path / "a,b"
        odd.mkdir()
        monkeypatch.setattr(mc_llm_paths, "data_root", lambda: odd)
        vector()
        cv.choose("happy.gguf")

        assert cv.launch_flags(Config(model)) == []

    def test_a_failed_start_sets_the_vector_aside_once(self, model, caplog):
        vector()
        cv.choose("happy.gguf")
        cv.launch_flags(Config(model))

        with caplog.at_level(logging.WARNING, logger="model_chain"):
            assert cv.set_aside_after_failure() is True
        assert cv.set_aside_after_failure() is False
        assert cv.launch_flags(Config(model)) == []
        assert cv.view()["state"] == "set_aside"
        assert "set aside" in caplog.text and "happy" not in caplog.text

        cv.choose("happy.gguf")
        assert cv.launch_flags(Config(model))[0] == cv.FLAG, "choosing again tries again"

    def test_with_no_vector_given_a_failure_is_not_the_vectors(self, model):
        cv.launch_flags(Config(model))

        assert cv.set_aside_after_failure() is False


class TestTheRuntime:
    def test_a_start_carries_the_flag_among_its_optional_ones(self, model, monkeypatch):
        vector()
        cv.choose("happy.gguf")
        monkeypatch.setattr(mc_llm_runtime, "accelerator_flags", lambda *args: [])
        monkeypatch.setattr(mc_llm_runtime, "_wangp_thread_flags", lambda *args: [])
        monkeypatch.setattr(mc_llm_runtime, "_template_flags", lambda *args: [])
        monkeypatch.setattr(mc_llm_runtime, "_saved_cache_flags", lambda *args: [])
        monkeypatch.setattr(mc_llm_runtime, "_with_runtime", lambda configuration, runtime:
                            configuration)

        flags = mc_llm_runtime._launch_flags(Config(model), None, None)

        assert flags[:1] == [cv.FLAG]
        assert cv.FLAG in mc_llm_runtime.OPTIONAL_FLAGS

    def test_a_failed_start_tries_again_without_the_vector_but_not_for_memory(self, model):
        vector()
        cv.choose("happy.gguf")
        cv.launch_flags(Config(model))

        class Failure:
            out_of_memory = True

        assert mc_llm_runtime._control_vector_set_aside(Failure()) is False
        Failure.out_of_memory = False
        assert mc_llm_runtime._control_vector_set_aside(Failure()) is True

    def test_a_saved_cache_of_one_vector_is_not_offered_to_another_or_none(self, model, tmp_path):
        runtime = tmp_path / "llama-server"
        runtime.write_text("x")
        vector()
        cv.choose("happy.gguf")
        steered = cv.launch_flags(Config(model))[1]

        import hashlib

        stamp = mc_llm_slot_cache._stamp
        before = hashlib.sha256("\n".join([stamp(runtime), stamp(model), stamp(None), "full",
                                           "default", stamp(None)]).encode("utf-8")
                                ).hexdigest()[:12]
        plain = mc_llm_slot_cache.identity(runtime, model, None, True)
        assert plain == before, "a server started as before keeps the caches it had"
        assert mc_llm_slot_cache.identity(runtime, model, None, True, steering="") == plain
        assert mc_llm_slot_cache.identity(runtime, model, None, True,
                                          steering=steered) != plain


class TestStartingTheModelAgain:
    def servers(self, monkeypatch, model):
        stopped = []

        class Server:
            def configuration(self):
                return Config(model)

            def stop(self):
                stopped.append(True)

        running = [Server()]
        monkeypatch.setattr(cv, "_servers_of", lambda found: list(running) if not stopped
                            else [])
        return stopped

    def test_a_change_stops_the_models_server_when_nothing_is_written(self, model, monkeypatch):
        stopped = self.servers(monkeypatch, model)
        monkeypatch.setattr(cv, "_busy", lambda: False)
        vector()

        cv.choose("happy.gguf")

        assert stopped == [True]

    def test_while_something_is_written_it_waits_and_then_stops_it(self, model, monkeypatch):
        stopped = self.servers(monkeypatch, model)
        busy = threading.Event()
        busy.set()
        monkeypatch.setattr(cv, "_busy", busy.is_set)
        monkeypatch.setattr(cv, "RESTART_POLL_SECONDS", 0.01)
        vector()

        cv.choose("happy.gguf")
        time.sleep(0.05)
        assert stopped == [] and cv.view()["state"] == "waiting"

        busy.clear()
        cv._waiting[cv._model_key(model)].join(timeout=5)
        assert stopped == [True]

    def test_the_same_choice_again_stops_nothing(self, model, monkeypatch):
        vector()
        cv.choose("happy.gguf")
        stopped = self.servers(monkeypatch, model)
        monkeypatch.setattr(cv, "_busy", lambda: False)

        cv.choose("happy.gguf")

        assert stopped == []


class TestAddingAVector:
    def test_a_vector_is_copied_in_under_a_safe_name_never_over_another(self, tmp_path):
        source = tmp_path / "made elsewhere.gguf"
        cv.write(source, {1: [1.0] * 8})

        first = cv.add(source, "../../My vector!.gguf")
        second = cv.add(source, "../../My vector!.gguf")

        assert first == "My vector.gguf" and second == "My vector-2.gguf"
        assert (cv.folder() / first).is_file() and (cv.folder() / second).is_file()

    def test_a_file_that_is_not_a_vector_is_refused(self, tmp_path):
        source = tmp_path / "notes.gguf"
        source.write_bytes(b"GGUFjunk")

        with pytest.raises(cv.Refused):
            cv.add(source)
        assert not cv.folder().exists() or not any(cv.folder().iterdir())


class TestWhatTheMenusShow:
    def test_off_the_vectors_with_whether_each_fits_and_no_path(self, model, data):
        vector()
        vector("wide.gguf", width=16)

        found = cv.view()

        assert found["state"] == "off" and found["vector"] == ""
        assert [(item["name"], item["fits"]) for item in found["vectors"]] == [
            ("happy.gguf", True), ("wide.gguf", False)]
        assert found["layers"] == 12 and (found["first_layer"], found["last_layer"]) == (4, 11)
        assert str(data) not in repr(found)

    def test_chosen_with_the_model_stopped_it_takes_effect_at_the_next_start(self, model):
        vector()
        cv.choose("happy.gguf")

        assert cv.view()["state"] == "next_start"

    def test_on_once_the_running_server_was_given_it(self, model, monkeypatch):
        vector()
        cv.choose("happy.gguf")
        cv.launch_flags(Config(model))
        monkeypatch.setattr(cv, "_servers_of", lambda found: [object()])

        assert cv.view()["state"] == "on"

    def test_a_chosen_vector_that_went_says_so(self, model):
        path = vector()
        cv.choose("happy.gguf")
        path.unlink()

        found = cv.view()

        assert found["state"] == "unusable" and "not in the folder" in found["said"]

    def test_with_no_model_it_says_to_set_one_up(self, monkeypatch):
        monkeypatch.setattr(mc_llm_runtime, "config", lambda role="": Config(None))

        assert cv.view()["state"] == "unavailable"
