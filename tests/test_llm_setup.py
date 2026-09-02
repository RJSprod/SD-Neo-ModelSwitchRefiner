"""Getting a llama.cpp runtime in place from inside Forge.

This exists because the panel could previously reach a state whose only
recovery instruction — inherited from the standalone application — named a Qt
wizard this extension does not have. The tests below are mostly about the three
routes out of that state, and about the containment rule that constrains all of
them: the runtime is a program this extension *starts*, so it has to live
inside the install root.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

import mc_llm_paths
import mc_llm_setup as setup


@pytest.fixture(autouse=True)
def root(tmp_path, monkeypatch, host):
    """A throwaway install root, and no cached device list between tests.

    A *subdirectory* of tmp_path, not tmp_path itself, so that a test writing
    an "existing llama.cpp build somewhere else on the machine" into tmp_path
    is really writing it outside the root. Rooting the install at tmp_path made
    every such build contained, and every containment assertion vacuous.
    """
    install = tmp_path / "install"
    install.mkdir()
    monkeypatch.setattr(mc_llm_paths, "data_root", lambda: install)
    setup.forget_devices()
    yield install
    setup.forget_devices()


@pytest.fixture
def elsewhere(tmp_path):
    """Somewhere genuinely outside the install root."""
    return tmp_path / "elsewhere"


@pytest.fixture
def a_card(monkeypatch):
    """One CUDA card in the machine, faked at the nvidia-smi boundary.

    Below the pairing rather than above it, so what the tests see is the list
    the panel sees: every card offered once holding the weights and once in
    mixed mode, with the processor after them.
    """
    from prompt_master.core.models import GpuInfo

    card = GpuInfo(0, "GPU-0000", "NVIDIA GeForce RTX 3090", 24576, 23304, "560.94", 8.6)
    monkeypatch.setattr("prompt_master.inference.device_detection.detect_gpus",
                        lambda *args, **kwargs: [card])
    setup.forget_devices()
    yield card
    setup.forget_devices()


def make_build(directory, name="llama-server", extras=("libllama.so", "libggml.so",
                                                       "llama-cli")):
    """A directory that looks like a llama.cpp release."""
    directory.mkdir(parents=True, exist_ok=True)
    server = directory / name
    server.write_bytes(b"#!/bin/sh\nexit 0\n")
    server.chmod(0o755)
    for extra in extras:
        (directory / extra).write_bytes(b"x")
    return server


class TestStatus:
    def test_a_bare_install_reports_nothing_ready(self, root):
        found = setup.status()

        assert not found.ready
        assert not found.adoptable
        assert found.recorded is None

    def test_a_build_in_place_but_unrecorded_is_adoptable(self, root):
        make_build(root / "runtime")

        found = setup.status()

        assert found.adoptable
        assert found.found is not None
        assert not found.ready

    def test_a_recorded_runtime_reads_as_ready(self, root):
        server = make_build(root / "runtime")

        setup.record(server)

        assert setup.status().ready

    def test_a_recorded_runtime_that_has_been_deleted_is_not_ready(self, root):
        server = make_build(root / "runtime")
        setup.record(server)
        server.unlink()

        assert not setup.status().ready

    def test_a_runtime_recorded_outside_the_root_is_refused(self, root, elsewhere):
        """The one path here that would start a program from somewhere this
        extension does not own."""
        from prompt_master.core.config import atomic_write_json

        outside = make_build(elsewhere / "bin")
        atomic_write_json(root / "data" / "setup-state.json", {"runtime": str(outside)})

        assert setup.recorded_runtime() is None

    def test_downloadability_is_reported_with_a_reason(self, root):
        found = setup.status()

        assert found.downloadable == (sys.platform == "win32")
        if not found.downloadable:
            assert "Windows" in found.detail
            assert "point the box" in found.detail.casefold()


class TestDetect:
    def test_it_finds_a_server_nested_under_the_runtime_directory(self, root):
        """Release archives extract into a subdirectory as often as not."""
        server = make_build(root / "runtime" / "build" / "bin")

        assert setup.detect() == server

    def test_it_finds_nothing_when_there_is_nothing(self, root):
        assert setup.detect() is None

    def test_it_ignores_things_that_are_not_a_server(self, root):
        directory = root / "runtime"
        directory.mkdir(parents=True)
        (directory / "readme.txt").write_text("hello", encoding="utf-8")

        assert setup.detect() is None


class TestAdopt:
    def test_a_build_already_inside_the_root_is_not_copied(self, root):
        """Copying it would leave two of a build the user deliberately placed."""
        server = make_build(root / "runtime")

        adopted, note = setup.adopt(server)

        assert adopted == server.resolve()
        assert "already in place" in note

    def test_a_release_directory_is_copied_whole(self, root, elsewhere):
        """llama.cpp loads shared libraries from beside the server, so taking
        the executable alone would produce a runtime that will not start."""
        make_build(elsewhere / "llama-b9637" / "bin")

        adopted, note = setup.adopt(elsewhere / "llama-b9637" / "bin" / "llama-server")

        assert (root / "runtime").is_dir()
        assert adopted.is_file()
        assert (adopted.parent / "libllama.so").is_file()
        assert "Copied the llama.cpp build" in note

    def test_a_directory_may_be_pointed_at_instead_of_the_executable(self, root, elsewhere):
        make_build(elsewhere / "release")

        adopted, _note = setup.adopt(elsewhere / "release")

        assert adopted.name == "llama-server"
        assert adopted.is_file()

    def test_a_system_binary_directory_yields_the_executable_alone(self, root, elsewhere):
        """Pointing at /usr/bin must not copy /usr/bin. The single-file route
        works, with a caveat the caller is told about."""
        system = elsewhere / "usr" / "bin"
        make_build(system, extras=("python3", "curl", "tar", "grep"))

        adopted, note = setup.adopt(system / "llama-server")

        assert adopted == root / "runtime" / "llama-server"
        assert not (root / "runtime" / "curl").exists()
        assert "only the executable was taken" in note

    def test_the_copied_executable_keeps_its_executable_bit(self, root, elsewhere):
        if sys.platform == "win32":
            pytest.skip("no executable bit on Windows")
        make_build(elsewhere / "release")

        adopted, _note = setup.adopt(elsewhere / "release")

        assert adopted.stat().st_mode & 0o111

    def test_a_second_adopt_replaces_the_first(self, root, elsewhere):
        make_build(elsewhere / "old", extras=("libllama.so", "libggml.so", "old-marker"))
        setup.adopt(elsewhere / "old")
        make_build(elsewhere / "new", extras=("libllama.so", "libggml.so", "new-marker"))

        setup.adopt(elsewhere / "new")

        assert (root / "runtime" / "new-marker").is_file()
        assert not (root / "runtime" / "old-marker").exists()

    def test_a_missing_path_says_so(self, root, tmp_path):
        with pytest.raises(setup.SetupError, match="nothing at"):
            setup.adopt(tmp_path / "nowhere")

    def test_a_directory_with_no_server_says_so(self, root, elsewhere):
        empty = elsewhere / "empty"
        empty.mkdir(parents=True)

        with pytest.raises(setup.SetupError, match="no llama-server"):
            setup.adopt(empty)

    def test_the_wrong_executable_is_named_rather_than_copied(self, root, elsewhere):
        elsewhere.mkdir(parents=True, exist_ok=True)
        other = elsewhere / "llama-cli"
        other.write_bytes(b"x")

        with pytest.raises(setup.SetupError, match="not a llama.cpp server"):
            setup.adopt(other)

    def test_an_implausibly_large_directory_is_refused(self, root, elsewhere, monkeypatch):
        """The guard against being pointed at a folder that contains a release
        rather than at the release."""
        monkeypatch.setattr(setup, "MAX_ADOPT_BYTES", 8)
        make_build(elsewhere / "release")

        with pytest.raises(setup.SetupError, match="larger than a llama.cpp release"):
            setup.adopt(elsewhere / "release")


class TestReplacingWhatIsAlreadyThere:
    """A replacement that cannot happen must leave the install as it was.

    Reported: pressing Download the pinned build over a runtime whose
    ``cublas64_12.dll`` was held open by a llama-server left from an earlier
    session did not fail cleanly -- the extractor clears its destination with
    ``rmtree`` first, and ``rmtree`` walks a folder file by file, so it deleted
    its way to the locked DLL and stopped. An install with a working llama.cpp
    before the press had none after it, and the panel came back with an empty
    runtime box.
    """

    def prepared(self, root):
        """A finished new build, and a runtime folder in place to be replaced."""
        runtime = root / "runtime"
        make_build(runtime)
        (runtime / "cublas64_12.dll").write_bytes(b"old")
        incoming = root / "incoming"
        make_build(incoming)
        (incoming / "cublas64_12.dll").write_bytes(b"new")
        return incoming, runtime

    def test_a_runtime_that_cannot_be_moved_aside_is_not_touched(self, root, monkeypatch):
        incoming, runtime = self.prepared(root)
        original = Path.rename

        def refuse(self, target):
            if self == runtime:
                raise PermissionError(
                    r"[WinError 5] Access is denied: 'runtime\\cublas64_12.dll'")
            return original(self, target)

        monkeypatch.setattr(Path, "rename", refuse)

        with pytest.raises(setup.SetupError, match="in use"):
            setup._replace_directory(incoming, runtime)

        assert (runtime / "llama-server").is_file()
        assert (runtime / "cublas64_12.dll").read_bytes() == b"old"

    def test_a_swap_that_fails_after_the_move_puts_the_old_one_back(self, root, monkeypatch):
        """The window where the runtime is neither the old one nor the new one.
        It is one rename wide, and it is the one that loses a build."""
        incoming, runtime = self.prepared(root)
        original = Path.rename

        def refuse(self, target):
            if self == incoming:
                raise PermissionError("[WinError 5] Access is denied")
            return original(self, target)

        monkeypatch.setattr(Path, "rename", refuse)

        with pytest.raises(setup.SetupError):
            setup._replace_directory(incoming, runtime)

        assert (runtime / "cublas64_12.dll").read_bytes() == b"old"
        assert setup.detect() is not None

    def test_a_replacement_that_works_leaves_the_new_build_and_no_leftovers(self, root):
        incoming, runtime = self.prepared(root)

        setup._replace_directory(incoming, runtime)

        assert (runtime / "cublas64_12.dll").read_bytes() == b"new"
        assert not incoming.exists()
        assert not (root / "runtime.previous").exists()

    def test_a_download_that_cannot_replace_the_runtime_keeps_it(self, root, monkeypatch):
        """The whole route, not just the swap: the archives are extracted
        beside the runtime, so the one in place is never opened until there is
        a complete build to put in its place."""
        import zipfile

        runtime = root / "runtime"
        make_build(runtime)
        archive = root / "build.zip"
        with zipfile.ZipFile(archive, "w") as bundle:
            bundle.writestr("llama-server", "#!/bin/sh\nexit 0\n")
            bundle.writestr("libllama.so", "x")
        monkeypatch.setattr(setup, "downloadable", lambda: True)
        component = types.SimpleNamespace(destination="runtime.zip")
        monkeypatch.setattr("prompt_master.provisioning.installer.load_components",
                            lambda: {"runtime": component})
        monkeypatch.setattr("prompt_master.provisioning.installer.runtime_component_ids",
                            lambda gpu: ["runtime"])
        monkeypatch.setattr("prompt_master.provisioning.downloader.download",
                            lambda component, destination, report: archive)
        original = Path.rename

        def refuse(self, target):
            if self == runtime:
                raise PermissionError(r"[WinError 5] Access is denied: 'cublas64_12.dll'")
            return original(self, target)

        monkeypatch.setattr(Path, "rename", refuse)

        with pytest.raises(setup.SetupError, match="in use"):
            setup.download()

        assert (runtime / "llama-server").is_file()
        assert not (root / ".runtime.incoming").exists()


class TestRecord:
    def test_choosing_mixed_mode_records_no_resident_layers(self, root, a_card):
        """The whole of the setting: the card is still named on --device, and
        the weights stay in system RAM."""
        server = make_build(root / "runtime")

        state = setup.record(server, setup.device_for_token("mixed:0"))

        assert state["mode"] == "mixed_aggressive"
        assert state["gpu_layers"] == "0"

    def test_choosing_the_card_itself_still_records_a_full_offload(self, root, a_card):
        server = make_build(root / "runtime")

        state = setup.record(server, setup.device_for_token("gpu:0"))

        assert state["mode"] == "gpu"
        assert state["gpu_layers"] == "all"

    def test_recording_a_runtime_again_leaves_a_mixed_install_mixed(self, root, a_card):
        """Detect answers "which llama-server", and nothing else. Answering it
        with "and put the weights back on the card" is how somebody who chose
        mixed mode ends up watching their VRAM fill."""
        server = make_build(root / "runtime")
        setup.record(server, setup.device_for_token("mixed:0"))

        state = setup.record(server)

        assert state["mode"] == "mixed_aggressive"
        assert state["gpu_layers"] == "0"

    def test_recording_for_a_role_writes_only_that_role(self, root, a_card):
        """Section 8: "Selecting B must not overwrite A." A role's card is
        recorded beside the installation's, never over it -- the installation is
        still what every other mode runs on and what the *other* role
        inherits."""
        import mc_llm_roles
        from prompt_master.core.config import read_json

        server = make_build(root / "runtime")
        setup.record(server, setup.device_for_token("gpu:0"))

        setup.record(server, setup.device_for_token("cpu:-1"),
                     role=mc_llm_roles.SPATIAL)

        state = read_json(mc_llm_paths.app_paths().state_file)
        assert state["mode"] == "gpu"
        assert state["roles"]["spatial"]["mode"] == "cpu"
        assert "creative" not in state.get("roles", {})

    def test_a_role_reads_back_the_device_it_was_given(self, root, a_card):
        import mc_llm_roles
        import mc_llm_runtime

        server = make_build(root / "runtime")
        setup.record(server, setup.device_for_token("gpu:0"))
        setup.record(server, setup.device_for_token("mixed_conservative:0"),
                     role=mc_llm_roles.CREATIVE)

        assert mc_llm_runtime.config().mode == "gpu"
        assert mc_llm_runtime.config(mc_llm_roles.CREATIVE).mode == "mixed_conservative"
        assert mc_llm_runtime.config(mc_llm_roles.SPATIAL).mode == "gpu"

    def test_forgetting_a_role_puts_it_back_on_the_installation(self, root, a_card):
        import mc_llm_roles
        import mc_llm_runtime

        server = make_build(root / "runtime")
        setup.record(server, setup.device_for_token("gpu:0"))
        setup.record(server, setup.device_for_token("cpu:-1"), role=mc_llm_roles.SPATIAL)
        assert mc_llm_runtime.config(mc_llm_roles.SPATIAL).mode == "cpu"

        setup.forget_role(mc_llm_roles.SPATIAL)

        assert mc_llm_runtime.config(mc_llm_roles.SPATIAL).mode == "gpu"

    def test_a_mixed_install_is_never_read_back_as_a_full_offload(self, root, a_card):
        """What every reader downstream turns into --n-gpu-layers."""
        import mc_llm_runtime

        server = make_build(root / "runtime")
        setup.record(server, setup.device_for_token("mixed:0"))

        configuration = mc_llm_runtime.config()

        assert configuration.gpu_layers == "0"
        assert not configuration.on_gpu

    def test_it_writes_a_usable_state_file_from_nothing(self, root):
        server = make_build(root / "runtime")

        state = setup.record(server)

        assert state["runtime"] == "runtime/llama-server"
        assert "gpu_index" in state and "gpu_device" in state
        assert (root / "data" / "setup-state.json").is_file()

    def test_the_runtime_is_recorded_relative_so_the_install_stays_movable(self, root):
        server = make_build(root / "runtime" / "bin")

        state = setup.record(server)

        assert not state["runtime"].startswith("/")
        assert state["runtime"] == "runtime/bin/llama-server"

    def test_an_existing_model_choice_survives_recording_a_runtime(self, root):
        """Fixing a missing runtime must not cost somebody the model they had
        already chosen."""
        from prompt_master.core.config import atomic_write_json

        atomic_write_json(root / "data" / "setup-state.json",
                          {"model": "models/mine.gguf", "mmproj": "models/proj.gguf"})
        server = make_build(root / "runtime")

        state = setup.record(server)

        assert state["model"] == "models/mine.gguf"
        assert state["mmproj"] == "models/proj.gguf"

    def test_a_runtime_outside_the_root_is_refused_with_the_reason(self, root, elsewhere):
        outside = make_build(elsewhere)

        with pytest.raises(setup.SetupError, match="outside the LLM data directory"):
            setup.record(outside)

    def test_a_missing_file_is_refused(self, root):
        with pytest.raises(setup.SetupError, match="no llama-server at"):
            setup.record(root / "runtime" / "absent")

    def test_recording_makes_the_config_usable(self, root):
        """The whole point: config().runtime stops being None, which is what
        the model chooser refuses on."""
        import mc_llm_runtime

        server = make_build(root / "runtime")
        setup.record(server)

        assert mc_llm_runtime.config().runtime == server.resolve()


class TestDevices:
    def test_a_device_list_is_always_offered(self, root):
        """Even with no nvidia-smi: the processor is a real answer, and it is
        what lets the panel offer CPU execution rather than nothing."""
        found = setup.devices()

        assert found

    def test_the_list_is_cached_between_calls(self, root, monkeypatch):
        """Detection is a subprocess with a fifteen-second timeout and the panel
        wants the answer several times while it is being built."""
        calls = []

        def counted(*args, **kwargs):
            calls.append(1)
            return []

        monkeypatch.setattr("prompt_master.inference.device_detection.detect_devices",
                            counted)
        setup.forget_devices()

        setup.devices()
        setup.devices()

        assert len(calls) == 1

    def test_a_refresh_asks_again(self, root, monkeypatch):
        calls = []
        monkeypatch.setattr("prompt_master.inference.device_detection.detect_devices",
                            lambda *a, **k: calls.append(1) or [])
        setup.forget_devices()

        setup.devices()
        setup.devices(refresh=True)

        assert len(calls) == 2

    def test_a_preferred_device_is_always_returned(self, root):
        assert setup.preferred_device() is not None

    def test_a_card_is_offered_three_ways_under_three_different_tokens(self, root, a_card):
        """The index alone is not an identity: the same card appears three
        times, and keyed on the index the mixed entries are indistinguishable
        from the entry that fills the card."""
        tokens = [setup.device_token(found) for found in setup.devices()]

        assert len(tokens) == len(set(tokens))
        assert {"gpu:0", "mixed_aggressive:0", "mixed_conservative:0"} <= set(tokens)

    def test_the_legacy_mixed_token_still_names_a_device(self, root, a_card):
        """Every installation configured before the split has "mixed:0" saved.
        Resolving it to nothing would drop the menu back to its first entry --
        the full offload — so an install told to keep the card free would
        quietly start filling it."""
        found = setup.device_for_token("mixed:0")

        assert found is not None
        assert found.mode == "mixed_aggressive"

    def test_a_token_resolves_to_the_device_it_names(self, root, a_card):
        assert setup.device_for_token("mixed_aggressive:0").is_mixed
        assert setup.device_for_token("mixed_conservative:0").is_conservative
        assert not setup.device_for_token("mixed_aggressive:0").is_conservative
        assert not setup.device_for_token("gpu:0").is_mixed
        assert setup.device_for_token("cpu:-1").is_cpu

    def test_a_bare_index_still_names_what_it_named_before(self, root, a_card):
        """What earlier builds wrote into the menu, and what it meant there:
        the first device with that index, card or processor."""
        assert not setup.device_for_token("0").is_mixed
        assert setup.device_for_token("-1").is_cpu

    def test_a_token_for_a_device_that_is_not_there_resolves_to_nothing(self, root, a_card):
        assert setup.device_for_token("gpu:7") is None
        assert setup.device_for_token("") is None


class TestDownload:
    def test_it_refuses_with_the_adopt_route_where_there_is_no_pinned_build(
            self, root, monkeypatch):
        monkeypatch.setattr(setup, "downloadable", lambda: False)

        with pytest.raises(setup.SetupError, match="Windows x64"):
            setup.download()

    def test_it_does_not_download_the_pinned_model(self, root, monkeypatch):
        """installer.provision would also fetch a 16-27 GiB model. Somebody who
        already has a GGUF wants the runtime and only the runtime."""
        monkeypatch.setattr(setup, "downloadable", lambda: True)
        requested = []

        def fake_fetch(component, destination, progress=None, notice=None):
            requested.append(component.component_id)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(b"")
            return destination

        monkeypatch.setattr("prompt_master.provisioning.downloader.download", fake_fetch)
        monkeypatch.setattr("prompt_master.provisioning.extractor.extract_zips_atomic",
                            lambda archives, destination: make_build(destination))

        setup.download()

        assert requested
        assert not any(key.startswith("model-") or key == "mmproj" for key in requested)


class TestADisprovedCardIsNotRecordedAnyway:
    """A build that reports an empty device list must not be recorded as CUDA0.

    The regression this covers took a working installation to a state where
    nothing could generate at all. A CUDA token was recorded while a runtime
    with no CUDA backend was on disk, and from then on every start passed
    ``--device CUDA0`` to a build that refuses it during argument parsing --
    before the model is opened, so the failure was identical on every model,
    every engine and every restart, and the retry loop's extra headroom could
    not touch it because nothing was ever short of memory.
    """

    LISTING_EMPTY = "Available devices:\n  (none)\n"
    LISTING_CARD = ("Available devices:\n  CUDA0: NVIDIA GeForce RTX 5090 "
                    "(32607 MiB, 31400 MiB free)\n")

    def _answers(self, monkeypatch, output, code=0):
        import subprocess

        class Result:
            returncode, stdout, stderr = code, output, ""

        monkeypatch.setattr(subprocess, "run", lambda *a, **k: Result())
        setup.forget_devices()

    def test_an_empty_listing_is_a_no(self, monkeypatch, root):
        server = make_build(root / "runtime")
        self._answers(monkeypatch, self.LISTING_EMPTY)

        assert setup.enumerates_a_device(server) is False

    def test_a_listed_card_is_a_yes(self, monkeypatch, root):
        server = make_build(root / "runtime")
        self._answers(monkeypatch, self.LISTING_CARD)

        assert setup.enumerates_a_device(server) is True

    def test_output_it_does_not_recognise_is_not_a_no(self, monkeypatch, root):
        """The important one. A probe that says nothing useful is not evidence
        that a machine has no GPU, and treating it as one would refuse every
        card the day llama.cpp rewords its listing."""
        server = make_build(root / "runtime")
        for output in ("", "ggml_backend_load_all: loading\n", "?"):
            self._answers(monkeypatch, output)
            assert setup.enumerates_a_device(server) is None, output

    def test_a_build_that_refused_the_flag_is_not_a_no(self, monkeypatch, root):
        server = make_build(root / "runtime")
        self._answers(monkeypatch, self.LISTING_EMPTY, code=1)

        assert setup.enumerates_a_device(server) is None

    def test_a_missing_build_is_not_a_no(self, root):
        assert setup.enumerates_a_device(root / "runtime" / "nothing-here") is None

    def test_it_is_asked_once_per_build(self, monkeypatch, root):
        import subprocess

        server = make_build(root / "runtime")
        asked = []

        class Result:
            returncode, stdout, stderr = 0, self.LISTING_CARD, ""

        monkeypatch.setattr(subprocess, "run",
                            lambda *a, **k: (asked.append(1), Result())[1])
        setup.forget_devices()
        for _ in range(4):
            assert setup.enumerates_a_device(server) is True
        assert len(asked) == 1, f"probed {len(asked)} times"

        # Replacing the build is what somebody does to fix this, so the answer
        # must not come out of the old build's cache.
        server.write_bytes(b"#!/bin/sh\nexit 0\n# rebuilt\n")
        assert setup.enumerates_a_device(server) is True
        assert len(asked) == 2

    def test_a_no_is_asked_again_rather_than_remembered(self, monkeypatch, root):
        """The asymmetry that matters, and it cost a user an hour of CPU inference.

        "This build lists a device" is a fact about the build and stays true
        while the file does not change. "It listed none" is not that kind of
        fact: this probe inherits the environment on purpose, so the answer
        depends on what the driver would show at that moment as well -- a
        transient driver state, a card held elsewhere, a CUDA_VISIBLE_DEVICES in
        the shell the WebUI was started from.

        Cached, one unlucky probe takes the card off every llama-server command
        line for the rest of the session, and a machine that was doing ninety
        tokens a second does fourteen with nothing to say it could recover.
        """
        import subprocess

        server = make_build(root / "runtime")
        asked, answer = [], [self.LISTING_EMPTY]

        class Result:
            returncode, stderr = 0, ""

            @property
            def stdout(self):
                return answer[0]

        monkeypatch.setattr(subprocess, "run",
                            lambda *a, **k: (asked.append(1), Result())[1])
        setup.forget_devices()

        assert setup.enumerates_a_device(server) is False
        assert setup.enumerates_a_device(server) is False
        assert len(asked) == 2, "a no is re-asked, not remembered"

        # And the moment the card is visible again, it is used again -- without
        # anybody restarting anything.
        answer[0] = self.LISTING_CARD
        assert setup.enumerates_a_device(server) is True
        assert len(asked) == 3

    def test_recording_a_card_this_build_cannot_see_is_refused(self, monkeypatch, root,
                                                               a_card):
        """Refused, so the working placement survives and the person is told why.

        Recording it anyway is what produced a machine that could not generate
        on any model or any engine.
        """
        server = make_build(root / "runtime")
        self._answers(monkeypatch, self.LISTING_EMPTY)

        with pytest.raises(setup.SetupError, match="no device to offload to"):
            setup.record(server, setup.device_for_token("gpu:0"))

    def test_a_probe_that_could_not_answer_still_falls_back(self, monkeypatch, root,
                                                            a_card):
        """The long-standing behaviour, kept: an unanswerable probe is not a no."""
        self._answers(monkeypatch, "nothing recognisable")
        server = make_build(root / "runtime")

        state = setup.record(server, setup.device_for_token("gpu:0"))

        assert state["gpu_device"] == "CUDA0"


class TestEachCardGetsTheBuildItCanRun:
    """CPU, a 3090 and a 5090 on one machine are all legitimate targets.

    Section 14 already allows the families to coexist -- ``runtime_directory``
    gives the second one its own directory precisely so a 5090 on CUDA 13 and a
    3090 on CUDA 12 can both be installed. What was missing was anybody reading
    that back when a device was recorded, so choosing a card beside the
    processor's build wrote a pair that cannot start: llama.cpp refuses
    ``--device CUDA0`` during argument parsing, before the model is opened, and
    every reply on every model and every engine dies the same way.
    """

    CPU = ("Intel(R) Core(TM) Ultra 9 185H", None)
    AMPERE = ("NVIDIA GeForce RTX 3090", 8.6)
    BLACKWELL = ("NVIDIA GeForce RTX 5090", 12.0)

    def _card(self, named, index=0):
        from prompt_master.core.models import GpuInfo

        name, compute = named
        if compute is None:
            return GpuInfo(-1, "", name, 0, 0, "", None)
        return GpuInfo(index, f"GPU-{index}", name, 24576, 23304, "560.94", compute)

    def _build(self, directory, family):
        server = make_build(directory)
        if family:
            (directory / setup.RUNTIME_MARKER).write_text(family, encoding="utf-8")
        return server.resolve()

    def test_a_card_is_recorded_against_its_own_family(self, root):
        """Both cards installed beside the processor's build; each finds its own."""
        cpu = self._build(root / "runtime", "llama-runtime-cpu")
        twelve = self._build(root / "runtime-llama-runtime-cuda12", "llama-runtime-cuda12")
        thirteen = self._build(root / "runtime-llama-runtime-cuda13", "llama-runtime-cuda13")
        paths = mc_llm_paths.app_paths()

        assert setup._runtime_for_device(paths, cpu, self._card(self.AMPERE, 1)) == twelve
        assert setup._runtime_for_device(paths, cpu, self._card(self.BLACKWELL)) == thirteen
        # And the one it was handed, when that is already right.
        assert setup._runtime_for_device(paths, twelve, self._card(self.AMPERE, 1)) == twelve

    def test_the_processor_runs_on_whichever_build_is_there(self, root):
        """``--device none`` is a CUDA build's business too.

        Demanding a separate download to run on the processor would be a new
        refusal where there was never a problem.
        """
        server = self._build(root / "runtime", "llama-runtime-cuda13")
        paths = mc_llm_paths.app_paths()

        assert setup._runtime_for_device(paths, server, self._card(self.CPU)) == server

    def test_a_card_with_no_build_is_refused_with_the_button_to_press(self, root):
        """Refused, not recorded. The message names the family and the remedy."""
        server = self._build(root / "runtime", "llama-runtime-cpu")
        paths = mc_llm_paths.app_paths()

        with pytest.raises(setup.SetupError, match="llama-runtime-cuda12"):
            setup._runtime_for_device(paths, server, self._card(self.AMPERE, 1))

    def test_an_unmarked_build_is_left_exactly_alone(self, root):
        """An install made before the marker existed is not a wrong family."""
        server = self._build(root / "runtime", "")
        paths = mc_llm_paths.app_paths()

        assert setup._runtime_for_device(paths, server, self._card(self.AMPERE, 1)) == server

    def test_recording_a_card_writes_its_family_and_its_build(self, root, a_card):
        """End to end: the pair that reaches the state file agrees with itself."""
        self._build(root / "runtime", "llama-runtime-cpu")
        twelve = self._build(root / "runtime-llama-runtime-cuda12", "llama-runtime-cuda12")

        state = setup.record(root / "runtime" / "llama-server",
                             setup.device_for_token("gpu:0"))

        assert state["runtime_id"] == "llama-runtime-cuda12"
        assert Path(state["runtime"]).name == twelve.name
        assert "cuda12" in str(state["runtime"]), state["runtime"]

    def test_mixed_mode_needs_the_card_s_build_just_as_much(self, root, a_card):
        """Mixed keeps the weights in system RAM and the card does the arithmetic.

        So it is a CUDA placement with no resident layers, not a CPU one, and a
        build that cannot see the card serves it no better than it serves a
        full offload.
        """
        self._build(root / "runtime", "llama-runtime-cpu")
        self._build(root / "runtime-llama-runtime-cuda12", "llama-runtime-cuda12")

        state = setup.record(root / "runtime" / "llama-server",
                             setup.device_for_token("mixed:0"))

        assert state["mode"] == "mixed_aggressive"
        assert state["gpu_layers"] == "0", "mixed must not become a full offload"
        assert state["runtime_id"] == "llama-runtime-cuda12"
        assert "cuda12" in str(state["runtime"]), state["runtime"]


class TestMixedMinimumIsOfferedBesideTheOthers:
    """A fourth way to use a card, without disturbing the other three.

    Mixed Minimum asks the opposite question from Aggressive -- not how much of
    the model fits in what the image side leaves, but how little is enough to
    keep real work on the card. For a mixture-of-experts backbone that is well
    defined: the experts go to system RAM and everything else stays resident.
    """

    def test_a_card_is_now_offered_four_ways(self, root, a_card):
        found = setup.devices()
        tokens = [setup.device_token(device) for device in found]

        assert "minimum:0" in tokens, tokens
        # And the three that existed are untouched, which is the requirement.
        assert "gpu:0" in tokens and "cpu:-1" in tokens
        assert any(token.startswith("mixed_aggressive") for token in tokens), tokens
        assert any(token.startswith("mixed_conservative") for token in tokens), tokens
        assert tokens.count("minimum:0") == 1, "the card was offered it twice"

    def test_it_is_never_offered_for_the_processor(self, root, monkeypatch):
        """There is no card to keep a minimum on."""
        monkeypatch.setattr("prompt_master.inference.device_detection.detect_gpus",
                            lambda *args, **kwargs: [])
        setup.forget_devices()

        assert not any(setup.is_minimum_device(device) for device in setup.devices())

    def test_the_token_resolves_to_a_minimum_card(self, root, a_card):
        found = setup.device_for_token("minimum:0")

        assert found is not None
        assert setup.is_minimum_device(found)
        assert not setup.is_minimum_device(setup.device_for_token("mixed_aggressive:0"))

    def test_every_lifecycle_rule_still_sees_aggressive(self, root, a_card):
        """The whole design, asserted rather than described.

        ``prompt_master`` is byte-identical to its upstream, so PLACEMENT_MODES
        cannot grow and ``normalise_mode`` answers "" to anything not in it. A
        mode this repository invented would read as *no mode at all* to every
        rule that asks -- residency, the broker's pool, eviction when the image
        side wants the card. Inheriting keeps every one of them correct.
        """
        from prompt_master.core.models import (GpuInfo, MIXED_AGGRESSIVE_MODE,
                                               normalise_mode)

        found = setup.device_for_token("minimum:0")

        assert isinstance(found, GpuInfo)
        assert found.mode == MIXED_AGGRESSIVE_MODE
        assert normalise_mode(found.mode) == MIXED_AGGRESSIVE_MODE
        assert found.is_mixed and not found.is_conservative
        assert found.weights_in_system_ram

    def test_the_choice_survives_a_restart(self, root, a_card):
        """Minimum and Aggressive share a mode, so this is the only thing that
        tells a restart which one was chosen. Without it the card somebody asked
        to keep nearly empty quietly fills up again."""
        server = make_build(root / "runtime")

        state = setup.record(server, setup.device_for_token("minimum:0"))

        assert state["expert_minimum"] is True
        assert state["mode"] == "mixed_aggressive"
        assert setup.is_minimum_device(setup.configured_device())

    def test_choosing_aggressive_clears_it_again(self, root, a_card):
        """Switching back must not leave the marker behind."""
        server = make_build(root / "runtime")
        setup.record(server, setup.device_for_token("minimum:0"))

        state = setup.record(server, setup.device_for_token("mixed_aggressive:0"))

        assert state["expert_minimum"] is False
        assert not setup.is_minimum_device(setup.configured_device())

    def test_its_menu_line_says_what_it_does_and_what_it_needs(self, root, a_card):
        line = setup.describe_device(setup.device_for_token("minimum:0"))

        assert "Mixed Minimum" in line
        assert "system RAM" in line
        # The limitation belongs on the label, not in a surprise later.
        assert "experts" in line.casefold()

