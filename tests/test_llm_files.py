"""Turning what somebody pasted into the file they meant.

Every case below is a real paste rather than an invented edge: a Windows *Copy
as path* with its quotes, a drag-and-drop that arrives as a URL, a folder
offered where a file was asked for, the wrong shard of a split model. The
module exists because each of them used to produce "there is nothing at ...",
naming a path that looks exactly right — which is about the least actionable
error message it is possible to produce.
"""

from __future__ import annotations

import sys

import pytest

import mc_llm_files as files


@pytest.fixture
def models(tmp_path):
    folder = tmp_path / "models"
    folder.mkdir()
    return folder


def make(path, size: int = 16):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x" * size)
    return path


# --------------------------------------------------------------------------- #
# What was typed
# --------------------------------------------------------------------------- #


class TestCleaning:
    def test_a_windows_copy_as_path_loses_its_quotes(self):
        assert files.clean('"C:\\models\\thing.gguf"') == "C:\\models\\thing.gguf"

    def test_a_single_stray_quote_goes_too(self):
        """A half-selected copy produces one quote and no closing one."""
        assert files.clean('"C:\\models\\thing.gguf') == "C:\\models\\thing.gguf"

    def test_smart_quotes_count_as_quotes(self):
        assert files.clean("\u201c/models/thing.gguf\u201d") == "/models/thing.gguf"

    def test_a_dragged_file_url_becomes_a_path(self):
        assert files.clean("file:///models/my%20models/thing.gguf") == \
            "/models/my models/thing.gguf"

    def test_a_windows_file_url_keeps_its_drive_letter(self):
        assert files.clean("file:///C:/models/thing.gguf") == "C:/models/thing.gguf"

    def test_environment_variables_are_expanded(self, monkeypatch):
        monkeypatch.setenv("MC_TEST_MODELS", "/srv/models")
        assert files.clean("$MC_TEST_MODELS/thing.gguf") == "/srv/models/thing.gguf"

    def test_a_wrapped_paste_keeps_the_first_line(self):
        """A path cannot contain a newline, so joining two would make neither."""
        assert files.clean("/models/one.gguf\n/models/two.gguf") == "/models/one.gguf"

    def test_a_trailing_separator_goes_but_a_root_survives(self):
        assert files.clean("/models/") == "/models"
        assert files.clean("/") == "/"
        assert files.clean("C:\\") == "C:\\"

    def test_zero_width_characters_from_a_rendered_document_are_dropped(self):
        assert files.clean("/models/\u200bthing.gguf") == "/models/thing.gguf"

    def test_nothing_typed_is_nothing_rather_than_a_path(self):
        assert files.to_path("   ") is None
        assert files.to_path(None) is None


# --------------------------------------------------------------------------- #
# What was meant
# --------------------------------------------------------------------------- #


class TestResolvingAModel:
    def test_a_quoted_path_to_a_real_file_resolves(self, models):
        made = make(models / "thing.gguf")

        assert files.resolve_model(f'"{made}"').path == made

    def test_a_folder_holding_one_model_resolves_to_it_and_says_so(self, models):
        made = make(models / "thing.gguf")

        found = files.resolve_model(models)

        assert found.path == made
        assert any("only model in that folder" in note for note in found.notes)

    def test_a_folder_holding_several_names_them_rather_than_guessing(self, models):
        make(models / "a.gguf")
        make(models / "b.gguf")

        with pytest.raises(files.PathError) as raised:
            files.resolve_model(models)

        assert "2 models" in str(raised.value)
        assert "a.gguf" in str(raised.value)

    def test_a_projector_beside_the_model_does_not_count_as_a_second_model(self, models):
        made = make(models / "thing.gguf")
        make(models / "mmproj-thing-f16.gguf")

        assert files.resolve_model(models).path == made

    def test_a_folder_of_folders_points_at_the_one_holding_models(self, models):
        make(models / "gemma" / "thing.gguf")

        with pytest.raises(files.PathError) as raised:
            files.resolve_model(models)

        assert "gemma" in str(raised.value)

    def test_an_empty_folder_says_so(self, models):
        with pytest.raises(files.PathError, match="no .gguf files"):
            files.resolve_model(models)

    def test_a_file_that_is_not_a_gguf_is_refused_by_name(self, models):
        made = make(models / "thing.safetensors")

        with pytest.raises(files.PathError, match="not a GGUF"):
            files.resolve_model(made)

    def test_a_typo_is_answered_with_the_file_that_is_there(self, models):
        make(models / "thing.gguf")

        with pytest.raises(files.PathError) as raised:
            files.resolve_model(models / "thnig.gguf")

        assert "did you mean" in str(raised.value).casefold()

    def test_a_wrong_drive_names_the_folder_that_is_missing(self, tmp_path):
        with pytest.raises(files.PathError) as raised:
            files.resolve_model(tmp_path / "nowhere" / "deeper" / "thing.gguf")

        assert "does not exist either" in str(raised.value)
        assert str(tmp_path / "nowhere") in str(raised.value)

    @pytest.mark.skipif(sys.platform == "win32",
                        reason="Windows matches capitalisation itself")
    def test_a_path_copied_off_a_windows_machine_matches_case_insensitively(self, models):
        made = make(models / "Thing.gguf")

        found = files.resolve_model(models / "thing.gguf")

        assert found.path == made
        assert any("capitalisation" in note for note in found.notes)

    def test_nothing_typed_asks_for_something(self, models):
        with pytest.raises(files.PathError, match="Browse"):
            files.resolve_model("")


class TestSplitModels:
    """llama.cpp is handed part one and finds the rest; handed part three it
    loads a third of a model and fails in a way nobody can read."""

    def test_a_later_shard_resolves_to_the_first_and_says_why(self, models):
        first = make(models / "big-00001-of-00003.gguf")
        third = make(models / "big-00003-of-00003.gguf")

        found = files.resolve_model(third)

        assert found.path == first
        assert any("part 1 of 3" in note for note in found.notes)

    def test_the_first_shard_passes_through_unremarked(self, models):
        first = make(models / "big-00001-of-00003.gguf")
        make(models / "big-00002-of-00003.gguf")

        found = files.resolve_model(first)

        assert found.path == first
        assert found.notes == ()

    def test_a_split_model_reads_as_one_model_in_a_folder(self, models):
        first = make(models / "big-00001-of-00003.gguf")
        make(models / "big-00002-of-00003.gguf")
        make(models / "big-00003-of-00003.gguf")

        assert files.resolve_model(models).path == first

    def test_a_shard_whose_first_part_is_absent_is_reported_not_hidden(self, models):
        third = make(models / "big-00003-of-00003.gguf")

        found = files.resolve_model(third)

        assert found.path == third
        assert any("every part in one folder" in note for note in found.notes)


class TestResolvingAProjector:
    def test_an_empty_box_is_a_text_only_model_rather_than_an_error(self):
        assert files.resolve_projector("", None) is None

    def test_a_projector_resolves_and_is_not_remarked_on(self, models):
        model = make(models / "thing.gguf")
        projector = make(models / "mmproj-thing-f16.gguf")

        found = files.resolve_projector(f'"{projector}"', model)

        assert found.path == projector
        assert found.notes == ()

    def test_a_folder_resolves_to_the_projector_in_it(self, models):
        model = make(models / "thing.gguf")
        projector = make(models / "mmproj-thing-f16.gguf")

        assert files.resolve_projector(models, model).path == projector

    def test_the_model_cannot_also_be_its_own_projector(self, models):
        model = make(models / "thing.gguf")

        with pytest.raises(files.PathError, match="cannot be the model file itself"):
            files.resolve_projector(model, model)

    def test_a_file_not_named_like_a_projector_is_warned_about_not_refused(self, models):
        model = make(models / "thing.gguf")
        other = make(models / "other.gguf")

        found = files.resolve_projector(other, model)

        assert found.path == other
        assert any("not named like a projector" in note for note in found.notes)


# --------------------------------------------------------------------------- #
# What is in a folder
# --------------------------------------------------------------------------- #


class TestListing:
    def test_it_separates_folders_from_matching_files(self, models):
        make(models / "thing.gguf")
        make(models / "notes.txt")
        (models / "sub").mkdir()

        found = files.listing(models)

        assert [path.name for path in found.folders] == ["sub"]
        assert [path.name for path in found.files] == ["thing.gguf"]

    def test_no_suffix_filter_lists_every_file(self, models):
        make(models / "llama-server")

        found = files.listing(models, suffixes=())

        assert [path.name for path in found.files] == ["llama-server"]

    def test_a_folder_that_is_not_there_falls_back_to_one_that_is(self, models):
        found = files.listing(models / "gone" / "deeper")

        assert found.directory == models.resolve()
        assert "is not a folder" in found.detail

    def test_an_unreadable_folder_lists_as_empty_rather_than_raising(self, models,
                                                                     monkeypatch):
        def refuse(_path):
            raise PermissionError("nope")

        monkeypatch.setattr(files.os, "scandir", refuse)

        assert files.listing(models).files == ()

    def test_hidden_entries_stay_hidden(self, models):
        make(models / ".secret.gguf")

        assert files.listing(models).files == ()

    def test_up_stops_at_the_root(self):
        from pathlib import Path

        root = Path(Path.cwd().anchor or "/")

        assert files.parent_of(root) == root


class TestWhereAPickerOpens:
    def test_it_opens_beside_whatever_is_already_in_the_box(self, models):
        made = make(models / "thing.gguf")

        assert files.starting_folder(str(made)) == models

    def test_a_folder_in_the_box_is_where_it_opens(self, models):
        assert files.starting_folder(str(models)) == models

    def test_an_empty_box_falls_back_to_what_the_caller_offered(self, models):
        assert files.starting_folder("", models) == models

    def test_the_places_list_only_holds_folders_that_exist(self):
        assert all(place.is_dir() for place in files.places())


class TestParityWithTheVendoredPackage:
    """These two constants are restated so this module answers on an install
    where the vendored package will not import. Restated is not forked."""

    def test_the_suffix_and_the_hints_still_match_upstream(self):
        from prompt_master.inference import model_choice

        assert files.MODEL_SUFFIX == model_choice.MODEL_SUFFIX
        assert files.PROJECTOR_HINTS == model_choice.PROJECTOR_HINTS
