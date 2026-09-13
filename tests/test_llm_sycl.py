"""Intel Arc through SYCL as an LLM runtime target.

Structured around section 15 of the design intent: device discovery, the role
and menu surfaces, runtime families, the launch contract, the broker's resource
domains, and the promise that an installation with no Intel graphics -- or one
set up before the field existed -- behaves exactly as it did.

Two facts run through every class here and are worth stating once. The Intel
GPU is a processor of its own, so a request on it competes with an image
generation for nothing. Its memory is the system's, so the weights it holds are
host RAM: counted there, admitted there, reclaimed there, and never declared as
VRAM on any card.
"""

from __future__ import annotations

import types
from pathlib import Path

import pytest

import mc_broker
import mc_gguf
import mc_llm_context as ctx
import mc_llm_paths
import mc_llm_roles as roles
import mc_llm_runtime as runtime
import mc_llm_runtime_components as pinned
import mc_llm_setup as setup
import mc_llm_sycl as sycl
from prompt_master.core.models import GpuInfo
from test_llm_context import build_model

_GB = 1024**3
_MB = 1024**2

ARC = "Intel(R) Arc(TM) Graphics"
LIMIT_MB = 55705  # 54.4 GB, the validation machine's Windows figure
ADAPTER_RAM_MB = 2048  # the field that must never become the ceiling

LISTING_ARC = ("Available devices:\n"
               "  SYCL0: Intel(R) Arc(TM) Graphics (55705 MiB, 41000 MiB free)\n")
LISTING_CUDA = ("Available devices:\n"
                "  CUDA0: NVIDIA GeForce RTX 3090 (24576 MiB, 23000 MiB free)\n")
ADAPTERS = ('[{"Name":"Intel(R) Arc(TM) Graphics","PNPDeviceID":"PCI\\\\VEN_8086&DEV_7D55&SUBSYS_1",'
            '"AdapterRAM":2147483648,"DriverVersion":"32.0.101.6"},'
            '{"Name":"NVIDIA GeForce RTX 3090","PNPDeviceID":"PCI\\\\VEN_10DE&DEV_2204",'
            '"AdapterRAM":4293918720,"DriverVersion":"560.94"}]')


@pytest.fixture(autouse=True)
def root(tmp_path, monkeypatch, host):
    """A throwaway install root and no cached device list of either kind."""
    install = tmp_path / "install"
    install.mkdir()
    monkeypatch.setattr(mc_llm_paths, "data_root", lambda: install)
    mc_broker.clear()
    setup.forget_devices()
    yield install
    setup.forget_devices()
    mc_broker.clear()


@pytest.fixture
def intel(monkeypatch):
    """One Intel Arc in the machine, faked at the OS boundary.

    Below the device list rather than above it, so what the tests see is what
    the panel sees: the adapter the OS reports, the limit the OS reports, and
    -- unless a test installs one -- no SYCL runtime yet.
    """
    adapter = sycl.IntelGraphics(ARC, "VEN_8086&DEV_7D55", ADAPTER_RAM_MB, "32.0.101.6")
    monkeypatch.setattr(sycl, "detect_intel_graphics", lambda timeout=0: [adapter])
    monkeypatch.setattr(sycl, "shared_limit_bytes", lambda pci_id="": LIMIT_MB * _MB)
    monkeypatch.setattr(sycl, "installed_runtime", lambda: None)
    setup.forget_devices()
    yield adapter
    setup.forget_devices()


@pytest.fixture
def a_card(monkeypatch):
    """One CUDA card beside it."""
    card = GpuInfo(0, "GPU-0000", "NVIDIA GeForce RTX 3090", 24576, 23304, "560.94", 8.6)
    monkeypatch.setattr("prompt_master.inference.device_detection.detect_gpus",
                        lambda *args, **kwargs: [card])
    setup.forget_devices()
    yield card
    setup.forget_devices()


def make_build(directory, family=""):
    """A directory that looks like a llama.cpp release, marked with a family."""
    directory.mkdir(parents=True, exist_ok=True)
    server = directory / "llama-server"
    server.write_bytes(b"#!/bin/sh\nexit 0\n")
    server.chmod(0o755)
    for extra in ("libllama.so", "libggml.so", "llama-cli"):
        (directory / extra).write_bytes(b"x")
    if family:
        (directory / setup.RUNTIME_MARKER).write_text(family, encoding="utf-8")
    return server.resolve()


def arc_device():
    return setup.device_for_token("sycl:0", setup.devices())


def listing(monkeypatch, text, code=0):
    """Make every subprocess -- the device probe included -- print ``text``."""
    import subprocess

    class Result:
        returncode, stdout, stderr = code, text, ""

    monkeypatch.setattr(subprocess, "run", lambda *a, **k: Result())


# --------------------------------------------------------------------------- #
# 15.1 Device discovery
# --------------------------------------------------------------------------- #


class TestAnIntelArcBecomesOneSyclTarget:
    def test_the_os_adapter_listing_is_read_for_intel_parts_only(self):
        found = sycl.parse_adapter_listing(ADAPTERS)

        assert [entry.name for entry in found] == [ARC]
        assert found[0].pci_id == "VEN_8086&DEV_7D55"
        assert found[0].adapter_ram_mb == ADAPTER_RAM_MB

    def test_one_adapter_is_one_device_target(self, intel):
        found = sycl.devices()

        assert len(found) == 1
        assert isinstance(found[0], GpuInfo)
        assert found[0].name == ARC
        assert found[0].mode == "gpu"
        assert not found[0].is_cpu and not found[0].is_mixed

    def test_the_token_is_backend_qualified(self, intel):
        device = sycl.devices()[0]

        assert setup.device_token(device) == "sycl:0"
        assert setup.device_for_token("sycl:0") is not None
        assert setup.device_for_token("gpu:0") is None

    def test_cuda0_and_sycl0_do_not_collide(self, intel, a_card):
        tokens = [setup.device_token(found) for found in setup.devices()]

        assert "gpu:0" in tokens and "sycl:0" in tokens
        assert setup.device_for_token("gpu:0").name.startswith("NVIDIA")
        assert setup.device_for_token("sycl:0").name == ARC

    def test_the_intel_target_sits_between_the_cards_and_the_processor(self, intel, a_card):
        found = setup.devices()

        assert not found[0].is_cpu and "NVIDIA" in found[0].name
        assert sycl.is_sycl_device(found[-2])
        assert found[-1].is_cpu

    def test_adapter_ram_is_not_the_ceiling(self, intel):
        """2 GB is the field Windows calls Adapter RAM. The target must be able
        to hold a model far larger than it, so it is recorded and used for
        nothing that decides capacity."""
        device = sycl.devices()[0]

        assert device.adapter_ram_mb == ADAPTER_RAM_MB
        assert device.shared_limit_mb == LIMIT_MB
        assert device.memory_total_mb == LIMIT_MB
        assert "2.0 GB" not in setup.describe_device(device)

    def test_the_shared_limit_is_reported_as_a_limit(self, intel):
        line = setup.describe_device(sycl.devices()[0])

        assert "SYCL" in line and "shared system memory" in line
        assert "up to 54.4 GB shared" in line
        assert "VRAM" not in line

    def test_an_unknown_limit_is_reported_as_unknown_rather_than_invented(
            self, intel, monkeypatch):
        monkeypatch.setattr(sycl, "shared_limit_bytes", lambda pci_id="": 0)
        setup.forget_devices()

        device = sycl.devices()[0]
        found = sycl.Budget(shared_limit=0, host_available=60 * _GB, host_reserve=2 * _GB)

        assert "up to" not in setup.describe_device(device)
        assert "Shared GPU limit: not reported by the OS" in sycl.detail_lines(found=found)
        assert found.safe == 58 * _GB

    def test_the_runtime_s_own_enumeration_is_the_launch_identity(self, intel, root,
                                                                   monkeypatch):
        """Once a SYCL build is installed, what it lists is what is recorded --
        including a name the OS spelled differently."""
        server = make_build(root / "runtime", pinned.SYCL_RUNTIME)
        monkeypatch.setattr(sycl, "installed_runtime", lambda: server)
        monkeypatch.setattr(sycl, "enumerate_devices", lambda executable, timeout=0: [
            sycl.ListedDevice("SYCL0", 0, "Intel(R) Arc(TM) Graphics 140V")])
        setup.forget_devices()

        assert sycl.devices()[0].name == "Intel(R) Arc(TM) Graphics 140V"

    def test_no_intel_graphics_means_no_intel_entry(self, monkeypatch, a_card):
        monkeypatch.setattr(sycl, "detect_intel_graphics", lambda timeout=0: [])
        monkeypatch.setattr(sycl, "installed_runtime", lambda: None)
        setup.forget_devices()

        assert not any(sycl.is_sycl_device(found) for found in setup.devices())
        assert sycl.devices() == []

    def test_the_device_listing_parser_reads_llama_cpp_s_own_line(self):
        found = sycl.parse_device_listing(LISTING_ARC)

        assert found == [sycl.ListedDevice("SYCL0", 0, ARC, 55705, 41000)]
        assert sycl.parse_device_listing(LISTING_CUDA) == []


class TestTheThreeMemoryNumbersStayApart:
    def test_the_safe_budget_is_the_smaller_of_the_two_ceilings(self):
        by_host = sycl.Budget(shared_limit=60 * _GB, host_available=30 * _GB,
                              host_reserve=2 * _GB)
        by_limit = sycl.Budget(shared_limit=20 * _GB, host_available=60 * _GB,
                               host_reserve=2 * _GB)

        assert by_host.safe == 28 * _GB
        assert by_limit.safe == 20 * _GB

    def test_what_our_own_servers_hold_comes_off_the_shared_side(self):
        found = sycl.Budget(shared_limit=20 * _GB, committed=15 * _GB,
                            host_available=60 * _GB, host_reserve=2 * _GB)

        assert found.shared_remaining == 5 * _GB
        assert found.safe == 5 * _GB

    def test_the_limit_is_never_read_as_free(self):
        """54.4 GB shared with 3 GB of host RAM free is 1 GB safe, not 54."""
        found = sycl.Budget(shared_limit=LIMIT_MB * _MB, host_available=3 * _GB,
                            host_reserve=2 * _GB)

        assert found.safe == 1 * _GB
        assert "54.4 GB shared GPU limit" in found.describe()

    def test_an_unreadable_host_is_unknown_not_zero(self):
        found = sycl.Budget(shared_limit=LIMIT_MB * _MB, host_available=0, known=False)

        assert "Safe for LLM now: unknown" in "\n".join(sycl.detail_lines(found=found))

    def test_the_shortfall_sentence_carries_both_numbers(self):
        found = sycl.Budget(shared_limit=LIMIT_MB * _MB, host_available=10 * _GB,
                            host_reserve=2 * _GB)

        said = sycl.shortfall_sentence(17 * _GB, found)

        assert "17.0 GB of shared system memory" in said
        assert "8.0 GB is currently safe" in said
        assert "2.0 GB host-RAM reserve" in said


# --------------------------------------------------------------------------- #
# 15.2 Roles and the menu
# --------------------------------------------------------------------------- #


class TestTheIntelTargetReachesEveryConfigureForScope:
    def test_the_role_override_list_carries_the_backend(self):
        assert "compute_backend" in roles.STATE_FIELDS

    def test_one_device_list_offers_it_for_the_installation_and_every_role(self, intel,
                                                                              a_card):
        import mc_llm_studio

        labels = {label for label, _value in mc_llm_studio._device_choices()}
        assert any(ARC in label and "shared system memory" in label for label in labels)
        for role in ("", *roles.ROLES):
            choices = mc_llm_studio._device_choices()
            assert "sycl:0" in [value for _label, value in choices], role
            assert mc_llm_studio._current_device(choices, role) is not None

    def test_recording_the_installation_on_intel_is_inherited_by_an_unsplit_role(
            self, intel, root, monkeypatch):
        server = make_build(root / "runtime", pinned.SYCL_RUNTIME)
        listing(monkeypatch, LISTING_ARC)

        setup.record(server, arc_device())

        assert runtime.config().uses_sycl_compute
        for role in roles.ROLES:
            found = runtime.config(role)
            assert found.uses_sycl_compute and found.device == "SYCL0", role

    def test_a_split_role_may_choose_intel_while_another_stays_on_cuda(
            self, intel, a_card, root, monkeypatch):
        cuda = make_build(root / "runtime", "llama-runtime-cuda12")
        intel_build = make_build(root / "runtime-llama-runtime-sycl", pinned.SYCL_RUNTIME)
        listing(monkeypatch, LISTING_CUDA)
        setup.record(cuda, setup.device_for_token("gpu:0"))
        listing(monkeypatch, LISTING_ARC)

        setup.record(intel_build, arc_device(), role=roles.SPATIAL)

        assert runtime.config().backend == "cuda"
        assert runtime.config(roles.CREATIVE).backend == "cuda"
        assert runtime.config(roles.SPATIAL).backend == "sycl"
        assert runtime.config(roles.SPATIAL).device == "SYCL0"
        assert runtime.config(roles.SPATIAL).gpu_layers == "all"

    def test_the_menu_shows_the_recorded_intel_device_for_its_scope(self, intel, a_card, root,
                                                                    monkeypatch):
        import mc_llm_studio

        cuda = make_build(root / "runtime", "llama-runtime-cuda12")
        intel_build = make_build(root / "runtime-llama-runtime-sycl", pinned.SYCL_RUNTIME)
        listing(monkeypatch, LISTING_CUDA)
        setup.record(cuda, setup.device_for_token("gpu:0"))
        listing(monkeypatch, LISTING_ARC)
        setup.record(intel_build, arc_device(), role=roles.SPATIAL)
        choices = mc_llm_studio._device_choices()

        assert mc_llm_studio._current_device(choices, "") == "gpu:0"
        assert mc_llm_studio._current_device(choices, roles.SPATIAL) == "sycl:0"
        assert mc_llm_studio._current_device(choices, roles.CREATIVE) == "gpu:0"

    def test_a_mixed_minimum_installation_reads_back_as_minimum_in_the_menu(self, a_card, root,
                                                                            monkeypatch):
        """Found on the way: the menu built ``mode:index`` and Minimum shares
        Aggressive's mode, so a Minimum installation showed as Aggressive."""
        import mc_llm_studio

        server = make_build(root / "runtime", "llama-runtime-cuda12")
        listing(monkeypatch, LISTING_CUDA)
        setup.record(server, setup.device_for_token("minimum:0"))

        assert mc_llm_studio._current_device(mc_llm_studio._device_choices(), "") == "minimum:0"

    def test_two_intel_roles_share_one_server_and_a_different_one_does_not(self, tmp_path,
                                                                           monkeypatch):
        intel_config = configured(tmp_path, device="SYCL0", compute_backend="sycl")
        cuda_config = configured(tmp_path, device="CUDA0")
        answers = {roles.CREATIVE: intel_config, roles.SPATIAL: cuda_config}
        monkeypatch.setattr(runtime, "config",
                            lambda role="": answers.get(role, intel_config))
        registry = runtime.RuntimeRegistry()

        assert registry.for_role(roles.CREATIVE) is registry.for_role(roles.NEUTRALIZER)
        assert registry.for_role(roles.CREATIVE) is not registry.for_role(roles.SPATIAL)


def configured(tmp_path, **over):
    """A resolved configuration with an empty model and runtime on disk."""
    server = tmp_path / "llama-server"
    server.write_bytes(b"")
    model = tmp_path / over.pop("model_name", "A.gguf")
    if not model.exists():
        model.write_bytes(b"")
    base = dict(runtime=server, model=model, mmproj=None, gpu_index=0, device="CUDA0",
                gpu_layers="all", context_size=8192, context_mode="fixed",
                context_buffer_gb=4.0, kv_type_k="f16", kv_type_v="f16", mode="gpu")
    base.update(over)
    return runtime.Config(**base)


# --------------------------------------------------------------------------- #
# 15.3 Runtime families
# --------------------------------------------------------------------------- #


class TestTheSyclRuntimeFamily:
    def test_the_sycl_archive_is_chosen_only_for_intel(self, intel, a_card):
        arc = arc_device()
        card = setup.device_for_token("gpu:0")
        cpu = setup.devices()[-1]

        assert pinned.runtime_component_id(arc) == "llama-runtime-sycl"
        assert pinned.runtime_component_id(card) == "llama-runtime-cuda12"
        assert pinned.runtime_component_id(cpu) == "llama-runtime-cpu"

    def test_no_cudart_companion_is_requested_for_sycl(self, intel, a_card):
        assert pinned.runtime_component_ids(arc_device()) == ("llama-runtime-sycl",)
        assert pinned.runtime_component_ids(setup.device_for_token("gpu:0")) == (
            "llama-runtime-cuda12", "llama-runtime-cuda12-cudart")
        assert pinned.runtime_component_ids(setup.devices()[-1]) == ("llama-runtime-cpu",)

    def test_the_exact_release_and_sha256_are_pinned(self):
        found = pinned.components()["llama-runtime-sycl"]

        assert found.url.endswith("/b10621/llama-b10621-bin-win-sycl-x64.zip")
        assert found.url.startswith("https://github.com/ggml-org/llama.cpp/releases/download/")
        assert found.sha256 == ("9744eb81396e5b52ecb85591c1c9a5e767de5ce8aedb741bd971d89e"
                                "14295004")
        assert found.version == "b10621-sycl"
        # The vendored families come through the same door, untouched.
        assert "llama-runtime-cuda12" in pinned.components()
        assert "llama-runtime-cpu" in pinned.components()

    def test_the_vendored_manifest_is_not_edited(self):
        import json

        manifest = json.loads((Path(__file__).resolve().parent.parent / "prompt_master"
                               / "release-manifest.json").read_text(encoding="utf-8"))

        assert not any("sycl" in item["component_id"] for item in manifest["components"])

    def test_the_families_coexist_in_their_own_directories(self, root):
        make_build(root / "runtime", "llama-runtime-cuda12")

        assert setup.runtime_directory("llama-runtime-sycl", root).name == (
            "runtime-llama-runtime-sycl")
        make_build(root / "runtime-llama-runtime-sycl", "llama-runtime-sycl")
        families = setup.runtime_families(root)

        assert set(families) == {"llama-runtime-cuda12", "llama-runtime-sycl"}

    def test_installing_sycl_leaves_another_role_s_runtime_in_place(self, intel, a_card, root,
                                                                     monkeypatch):
        cuda = make_build(root / "runtime", "llama-runtime-cuda12")
        before = cuda.read_bytes()
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

        installed = setup.download(arc_device())

        assert requested == ["llama-runtime-sycl"]
        assert installed.parent.name == "runtime-llama-runtime-sycl"
        assert setup.family_in(installed.parent) == "llama-runtime-sycl"
        assert cuda.read_bytes() == before
        assert setup.family_in(root / "runtime") == "llama-runtime-cuda12"

    def test_recording_intel_against_a_cuda_only_install_names_the_button(self, intel, root):
        server = make_build(root / "runtime", "llama-runtime-cuda12")

        with pytest.raises(setup.SetupError, match="Intel Arc is available"):
            setup._runtime_for_device(mc_llm_paths.app_paths(), server, arc_device())

    def test_recording_intel_finds_the_sycl_family_beside_a_cuda_one(self, intel, root):
        cuda = make_build(root / "runtime", "llama-runtime-cuda12")
        intel_build = make_build(root / "runtime-llama-runtime-sycl", pinned.SYCL_RUNTIME)

        assert setup._runtime_for_device(mc_llm_paths.app_paths(), cuda,
                                         arc_device()) == intel_build


class TestRecordingRefusesWhatTheRuntimeCannotSee:
    def test_a_build_that_lists_no_sycl_device_is_refused(self, intel, root, monkeypatch):
        server = make_build(root / "runtime", pinned.SYCL_RUNTIME)
        listing(monkeypatch, "Available devices:\n  (none)\n")

        with pytest.raises(setup.SetupError, match="could not see Intel Arc"):
            setup.record(server, arc_device())
        assert not (root / "data" / "setup-state.json").exists()

    def test_a_build_that_lists_only_cuda_is_refused(self, intel, root, monkeypatch):
        server = make_build(root / "runtime", pinned.SYCL_RUNTIME)
        listing(monkeypatch, LISTING_CUDA)

        with pytest.raises(setup.SetupError, match="could not see Intel Arc"):
            setup.record(server, arc_device())

    def test_a_build_that_cannot_be_asked_is_refused_rather_than_assumed(self, intel, root,
                                                                         monkeypatch):
        server = make_build(root / "runtime", pinned.SYCL_RUNTIME)
        listing(monkeypatch, "", code=1)

        with pytest.raises(setup.SetupError, match="could not be asked"):
            setup.record(server, arc_device())

    def test_a_device_that_is_no_longer_enumerated_is_refused(self, intel, root, monkeypatch):
        server = make_build(root / "runtime", pinned.SYCL_RUNTIME)
        listing(monkeypatch, "Available devices:\n  SYCL1: Intel(R) UHD Graphics\n")

        with pytest.raises(setup.SetupError, match="no longer enumerated"):
            setup.record(server, arc_device())

    def test_a_device_that_is_plainly_something_else_is_refused(self, intel, root, monkeypatch):
        server = make_build(root / "runtime", pinned.SYCL_RUNTIME)
        listing(monkeypatch, "Available devices:\n  SYCL0: AMD Radeon 780M\n")

        with pytest.raises(setup.SetupError, match="not the Intel"):
            setup.record(server, arc_device())

    def test_a_real_device_token_is_what_reaches_the_state_file(self, intel, root, monkeypatch):
        server = make_build(root / "runtime", pinned.SYCL_RUNTIME)
        listing(monkeypatch, LISTING_ARC)

        state = setup.record(server, arc_device())

        assert state["gpu_device"] == "SYCL0"
        assert state["gpu_device_name"] == ARC
        assert state["compute_backend"] == "sycl"
        assert state["runtime_id"] == "llama-runtime-sycl"
        assert state["gpu_layers"] == "all"
        assert state["mode"] == "gpu"
        assert state["gpu_index"] == 0
        assert "shared_limit" not in state and "54" not in str(state.get("gpu_name"))

    def test_the_recorded_device_reads_back_as_intel(self, intel, root, monkeypatch):
        server = make_build(root / "runtime", pinned.SYCL_RUNTIME)
        listing(monkeypatch, LISTING_ARC)
        setup.record(server, arc_device())

        found = setup.configured_device()

        assert sycl.is_sycl_device(found)
        assert setup.device_token(found) == "sycl:0"

    def test_cuda_and_cpu_recording_write_their_backend_too(self, a_card, root, monkeypatch):
        server = make_build(root / "runtime", "llama-runtime-cuda12")
        listing(monkeypatch, LISTING_CUDA)

        assert setup.record(server, setup.device_for_token("gpu:0"))["compute_backend"] == "cuda"
        assert setup.record(server, setup.devices()[-1])["compute_backend"] == "cpu"


# --------------------------------------------------------------------------- #
# 15.4 The launch contract
# --------------------------------------------------------------------------- #


class FakeProcess:
    """A llama-server that starts instantly and holds nothing."""

    def __init__(self, started: list):
        self._started = started
        self.port = 8080
        self.api_key = "test"
        self.alive = False

    def start(self, *args, **kwargs):
        self._started.append((args, kwargs))
        self.alive = True

    def wait_ready(self, timeout=0):
        return None

    def stop(self):
        self.alive = False

    @property
    def running(self) -> bool:
        return self.alive


@pytest.fixture
def placed(monkeypatch, tmp_path):
    """A register nobody has written into, and a start that measures nothing slow."""
    monkeypatch.setattr(ctx, "_store_path", lambda: tmp_path / "calibration.json")
    ctx.forget()
    for family in (mc_broker.FAMILY_IMAGE, mc_broker.FAMILY_LLM):
        mc_broker.unregister_reclaimer(family)
    monkeypatch.setattr(mc_broker, "safety_margin_bytes", lambda: 0)
    monkeypatch.setattr(runtime, "_prime_prompt_cache", lambda client: None)
    monkeypatch.setattr(runtime, "RESIDENCY_SETTLE_SECONDS", 0.0)
    monkeypatch.setattr(runtime, "OFFLOAD_WAIT_SECONDS", 0.0)
    monkeypatch.setattr(runtime, "_runtime_enumerates_a_device", lambda executable: True)
    yield
    ctx.forget()


@pytest.fixture
def server(monkeypatch, tmp_path):
    started: list = []
    managed = runtime.Runtime()
    monkeypatch.setattr(managed, "_new_process", lambda: FakeProcess(started))
    yield managed, started
    managed.stop()


def configure_intel(monkeypatch, tmp_path, *, size_mb=64, context=8192, mode="fixed",
                    priority="cooperative"):
    """An installation on the Intel GPU, with a real GGUF header to size."""
    model = build_model(tmp_path, blocks=32, size_mb=size_mb, context=131072)
    binary = tmp_path / "llama-server"
    binary.write_bytes(b"")
    configuration = runtime.Config(
        runtime=binary, model=model, mmproj=None, gpu_index=0, device="SYCL0",
        gpu_layers="all", context_size=context, context_mode=mode, context_buffer_gb=4.0,
        kv_type_k="f16", kv_type_v="f16", mode="gpu", device_name=ARC, gpu_name=ARC,
        compute_backend="sycl", memory_priority=priority)
    monkeypatch.setattr(runtime, "config", lambda role="": configuration)
    return configuration


def weighing(monkeypatch, gigabytes):
    """Make the header say the file is this large, without writing it.

    The estimator sizes the weights from the header's own record of the file,
    and a seventeen-gigabyte file is neither writable in a test nor needed:
    what is under test is the arithmetic, and the arithmetic reads a number.
    """
    import dataclasses

    original = mc_gguf.describe

    def described(model, *args, **kwargs):
        found = original(model, *args, **kwargs)
        return dataclasses.replace(found, file_bytes=int(gigabytes * _GB)) if found else found

    monkeypatch.setattr(mc_gguf, "describe", described)


def budget_of(monkeypatch, host_gb, limit_gb=54.4, reserve_gb=2.0):
    found = sycl.Budget(shared_limit=int(limit_gb * _GB), host_available=int(host_gb * _GB),
                        host_reserve=int(reserve_gb * _GB))
    monkeypatch.setattr(sycl, "budget", lambda *args, **kwargs: found)
    monkeypatch.setattr(mc_broker, "free_ram_bytes", lambda: int(host_gb * _GB))
    monkeypatch.setattr(mc_broker, "host_ram_fits", lambda needed, reserve=None: True)
    return found


class TestTheCommandThatStartsAnIntelServer:
    def test_the_command_selects_the_sycl_device_with_every_layer(self, placed, server,
                                                                   tmp_path, monkeypatch):
        managed, started = server
        configure_intel(monkeypatch, tmp_path)
        budget_of(monkeypatch, 40)

        managed.client()

        (executable, model, projector, index, device, context, log), options = started[0]
        assert device == "SYCL0"
        assert index == 0
        assert options["gpu_layers"] == "33"  # 32 blocks and the output layer
        assert managed.placement().uma
        assert managed.placement().gpu_layers == ctx.ALL_LAYERS

    def test_the_backend_is_armed_for_the_launcher_and_consumed_by_it(self, placed, server,
                                                                       tmp_path, monkeypatch):
        managed, started = server
        configure_intel(monkeypatch, tmp_path)
        budget_of(monkeypatch, 40)
        seen = []
        original = runtime.with_backend_isolation

        def spy(command, environ):
            seen.append(list(runtime._pending_backend))
            return original(command, environ)

        monkeypatch.setattr(runtime, "with_backend_isolation", spy)

        managed.client()

        # Armed before the start; the fake process never reaches the launcher,
        # so it is still armed here and must be consumed by the next real one.
        assert runtime._pending_backend == ["sycl"]
        runtime._arm_backend("")

    def test_cuda_visible_devices_is_not_the_sycl_selector(self):
        """The vendored launcher wrote the device index into it. For SYCL that
        index is a SYCL ordinal being read as an NVIDIA slot."""
        command = ["llama-server", "--model", "m.gguf", "--ctx-size", "8192",
                   "--device", "SYCL0"]
        runtime._arm_backend("sycl")

        found = runtime.with_backend_isolation(command, {"CUDA_VISIBLE_DEVICES": "0",
                                                         "PATH": "x"})

        assert found["CUDA_VISIBLE_DEVICES"] == ""
        assert found["PATH"] == "x"
        assert runtime._pending_backend == [], "consumed by the start that armed it"

    def test_a_cuda_launch_keeps_its_environment(self):
        command = ["llama-server", "--model", "m.gguf", "--ctx-size", "8192",
                   "--device", "CUDA0"]
        runtime._arm_backend("cuda")

        assert runtime.with_backend_isolation(command, {"CUDA_VISIBLE_DEVICES": "0"}) == {
            "CUDA_VISIBLE_DEVICES": "0"}

    def test_a_cpu_launch_keeps_its_environment(self):
        command = ["llama-server", "--model", "m.gguf", "--ctx-size", "8192",
                   "--device", "none"]
        runtime._arm_backend("cpu")

        assert runtime.with_backend_isolation(command, {"CUDA_VISIBLE_DEVICES": ""}) == {
            "CUDA_VISIBLE_DEVICES": ""}

    def test_a_probe_spawned_during_a_start_is_left_alone(self):
        runtime._arm_backend("sycl")
        probe = ["nvidia-smi", "--query-gpu=index"]

        assert runtime.with_backend_isolation(probe, {"CUDA_VISIBLE_DEVICES": "0"}) == {
            "CUDA_VISIBLE_DEVICES": "0"}
        runtime._arm_backend("")

    def test_through_the_real_launcher_the_process_sees_no_nvidia_card(self, tmp_path,
                                                                       monkeypatch):
        """End to end through the vendored launcher and the wrapper in front of it."""
        import subprocess

        from prompt_master.inference import llama_process

        runtime.runtime._new_process()  # installs the wrapper, as a start does
        seen = {}

        class FakePopen:
            def __init__(self, command, *args, **kwargs):
                seen["command"] = [str(part) for part in command]
                seen["env"] = kwargs.get("env")
                self.args = command

            def poll(self):
                return None

            def terminate(self):
                pass

            def wait(self, timeout=None):
                return 0

        monkeypatch.setattr(subprocess, "Popen", FakePopen)
        monkeypatch.setattr(runtime, "_runtime_enumerates_a_device", lambda executable: True)
        runtime._arm_backend("sycl")
        process = llama_process.LlamaProcess()
        process.start(tmp_path / "llama-server", tmp_path / "model.gguf", None,
                      0, "SYCL0", 8192, tmp_path / "log.txt", gpu_layers="33")
        process.process = None

        assert seen["env"]["CUDA_VISIBLE_DEVICES"] == ""
        assert seen["command"][seen["command"].index("--device") + 1] == "SYCL0"
        assert "--n-gpu-layers" in seen["command"]

    def test_a_selected_sycl_target_never_silently_becomes_cpu(self, placed, server, tmp_path,
                                                               monkeypatch):
        """The launcher drops a device selection a build cannot enumerate and
        starts on the processor. Right for a CUDA token recorded beside a
        CPU-only build; wrong for an Intel target somebody chose by name."""
        managed, started = server
        configure_intel(monkeypatch, tmp_path)
        budget_of(monkeypatch, 40)
        monkeypatch.setattr(runtime, "_runtime_enumerates_a_device", lambda executable: False)

        with pytest.raises(RuntimeError, match="could not see Intel Arc"):
            managed.client()
        assert not started

    def test_a_model_far_larger_than_adapter_ram_is_admitted_when_ram_permits(
            self, placed, server, tmp_path, monkeypatch):
        """Acceptance test A7: the 2 GB field is not the ceiling."""
        managed, started = server
        configure_intel(monkeypatch, tmp_path)
        weighing(monkeypatch, 17)
        budget_of(monkeypatch, host_gb=40)

        managed.client()

        assert started
        assert managed.report.fits
        assert managed.report.estimate.weights_bytes == 17 * _GB

    def test_a_request_beyond_the_safe_budget_is_refused_with_both_numbers(
            self, placed, server, tmp_path, monkeypatch):
        """Acceptance test A11: below the OS limit and still refused, because
        the host does not have it."""
        managed, started = server
        configure_intel(monkeypatch, tmp_path)
        weighing(monkeypatch, 17)
        budget_of(monkeypatch, host_gb=6)

        with pytest.raises(RuntimeError) as refused:
            managed.client()

        said = str(refused.value)
        assert "GB of shared system memory" in said
        assert "4.0 GB is currently safe" in said
        assert "2.0 GB host-RAM reserve" in said
        assert not started

    def test_the_context_gives_ground_before_the_request_is_refused(self, placed, tmp_path,
                                                                     monkeypatch):
        configuration = configure_intel(monkeypatch, tmp_path, size_mb=64, context=65536)
        model = mc_gguf.read(configuration.model)
        per_token = ctx.kv_bytes_per_token(model, ctx.Placement(context=65536))
        # Room for the weights, the compute buffer and about 16K tokens.
        budget_of(monkeypatch, host_gb=(64 * _MB + 400 * _MB + per_token * 16384) / _GB + 2)

        negotiated = runtime.negotiate(configuration)

        assert negotiated.fits
        assert negotiated.placement.uma
        assert 8192 <= negotiated.placement.context < 65536
        assert any("context reduced" in note and "shared system memory" in note
                   for note in negotiated.notes)

    def test_an_unreadable_host_admits_rather_than_refuses(self, placed, tmp_path, monkeypatch):
        configure_intel(monkeypatch, tmp_path)
        monkeypatch.setattr(sycl, "budget", lambda *a, **k: sycl.Budget(known=False))

        negotiated = runtime.negotiate(runtime.config())

        assert negotiated.fits
        assert any("could not be read" in note for note in negotiated.notes)

    def test_the_start_line_names_the_shared_memory_rather_than_a_card(self, placed, server,
                                                                        tmp_path, monkeypatch,
                                                                        caplog):
        managed, _started = server
        configure_intel(monkeypatch, tmp_path)
        budget_of(monkeypatch, 40)
        asked = []
        monkeypatch.setattr(mc_broker, "device_free_vram_bytes",
                            lambda index=None: asked.append(index) or 20 * _GB)

        with caplog.at_level("INFO", logger="model_chain"):
            managed.client()

        start = next(r.getMessage() for r in caplog.records if "starting llama-server" in
                     r.getMessage())
        ready = next(r.getMessage() for r in caplog.records if "llama-server ready" in
                     r.getMessage())
        assert "shared system memory safe to use" in start
        assert "all layers on the Intel GPU, in shared system memory" in ready
        assert "GB VRAM" not in ready
        assert "GB of shared system memory" in ready
        assert asked == [], "no card's free VRAM is read for an Intel placement"

    def test_flash_attention_is_asked_for_not_insisted_on(self, tmp_path, monkeypatch):
        runtime._capabilities.clear()
        binary = tmp_path / "llama-server"
        binary.write_text("")
        monkeypatch.setattr(runtime.subprocess, "run", lambda *a, **k: types.SimpleNamespace(
            stdout="  -fa, --flash-attn [on|off|auto]  set Flash Attention use ('on', 'off', "
                   "or 'auto', default: 'auto')\n", stderr=""))
        configuration = runtime.Config(
            runtime=binary, model=tmp_path / "m.gguf", mmproj=None, gpu_index=0,
            device="SYCL0", gpu_layers="all", context_size=8192, context_mode="fixed",
            context_buffer_gb=4.0, kv_type_k="f16", kv_type_v="f16")

        flags = runtime.accelerator_flags(configuration, ctx.Placement(uma=True))

        assert flags[-2:] == ["--flash-attn", "auto"]
        runtime._capabilities.clear()


# --------------------------------------------------------------------------- #
# 15.5 The broker: execution domains and memory domains
# --------------------------------------------------------------------------- #


class TestTheIntelGpuIsItsOwnProcessor:
    def test_the_truth_table(self):
        intel0, intel1 = mc_broker.sycl_execution(0), mc_broker.sycl_execution(1)
        cuda0 = mc_broker.cuda_execution(0)

        assert not mc_broker.CPU_EXECUTION.conflicts_with(intel0)
        assert not intel0.conflicts_with(mc_broker.CPU_EXECUTION)
        assert not cuda0.conflicts_with(intel0)
        assert not intel0.conflicts_with(cuda0)
        assert not mc_broker.UNKNOWN_CUDA_EXECUTION.conflicts_with(intel0)
        assert intel0.conflicts_with(mc_broker.sycl_execution(0))
        assert not intel0.conflicts_with(intel1)
        assert cuda0.conflicts_with(mc_broker.cuda_execution(0))

    def test_an_unnamed_intel_device_conflicts_with_every_intel_device_and_nothing_else(self):
        unknown = mc_broker.sycl_execution(None)

        assert unknown.conflicts_with(mc_broker.sycl_execution(3))
        assert not unknown.conflicts_with(mc_broker.cuda_execution(3))
        assert not unknown.known

    def test_a_sycl_configuration_resolves_to_its_own_domain(self, tmp_path):
        found = runtime.execution_domain(configured(tmp_path, device="SYCL0", gpu_index=0))

        assert found == mc_broker.sycl_execution(0)
        assert found.is_sycl and found.known
        assert found.describe() == "the Intel GPU (SYCL0)"

    def test_a_cuda_configuration_resolves_exactly_as_before(self, tmp_path):
        assert runtime.execution_domain(configured(tmp_path)) == mc_broker.cuda_execution(0)
        assert runtime.execution_domain(configured(tmp_path, device="none", mode="cpu")) == (
            mc_broker.CPU_EXECUTION)

    def test_an_image_job_on_nvidia_does_not_wait_for_an_intel_llm(self):
        with mc_broker.workload(mc_broker.FAMILY_LLM, "a conversation reply", timeout=0,
                                domain=mc_broker.sycl_execution(0)) as held:
            assert held
            assert mc_broker.conflicting_llm(mc_broker.cuda_execution(1)) is None
            assert mc_broker.await_idle(0.05, domain=mc_broker.cuda_execution(1))
            # And a second Intel request still takes its turn.
            assert mc_broker.conflicting_llm(mc_broker.sycl_execution(0)) is not None

    def test_the_intel_llm_does_not_wait_for_an_nvidia_image_job(self, tmp_path, monkeypatch):
        import mc_llm_sessions

        monkeypatch.setattr(mc_broker, "host_busy", lambda: True)
        monkeypatch.setattr(mc_broker, "image_execution_domain",
                            lambda: mc_broker.cuda_execution(1))

        assert not mc_llm_sessions._Gpu._blocked_by_the_image_job(mc_broker.sycl_execution(0))
        assert mc_llm_sessions._Gpu._blocked_by_the_image_job(mc_broker.cuda_execution(1))


class TestIntelMemoryIsHostRam:
    def test_the_placement_is_charged_to_host_ram_in_full(self, tmp_path):
        configuration = configured(tmp_path, device="SYCL0", compute_backend="sycl")
        configuration.model.write_bytes(b"x" * 4096)
        placement = ctx.Placement(gpu_layers=ctx.ALL_LAYERS, on_gpu=True, uma=True)

        assert runtime.host_ram_demand(configuration, placement) == 4096
        assert runtime.host_ram_demand(configuration, None) == 4096

    def test_a_running_intel_server_declares_no_vram_anywhere(self, placed, server, tmp_path,
                                                              monkeypatch):
        managed, _started = server
        configure_intel(monkeypatch, tmp_path)
        budget_of(monkeypatch, 40)

        managed.client()

        assert managed.running()
        assert managed.resident_bytes() == 0
        assert mc_broker.resident_bytes(mc_broker.FAMILY_LLM) == 0
        assert managed.host_ram_bytes() == managed.configuration().model.stat().st_size
        assert managed.host_ram_claim()[1] == managed.host_ram_bytes()

    def test_it_is_never_a_vram_victim_for_any_card(self, placed, server, tmp_path, monkeypatch):
        managed, _started = server
        configure_intel(monkeypatch, tmp_path)
        budget_of(monkeypatch, 40)
        managed.client()

        assert not managed.on_card(0)
        assert not managed.on_card(1)
        assert managed.on_card(mc_broker.ANY_CARD)

    def test_an_intel_request_never_evicts_nvidia_vram_even_under_llm_priority(
            self, placed, tmp_path, monkeypatch):
        configuration = configure_intel(monkeypatch, tmp_path, priority="llm_priority")
        asked = []
        monkeypatch.setattr(mc_broker, "release_for_llm",
                            lambda *args, **kwargs: asked.append(args) or None)

        assert runtime._make_room_for_the_llm(configuration, 0, False, 0) == 0
        assert asked == []

    def test_the_image_plan_does_not_apply_to_an_intel_server(self, placed, tmp_path,
                                                              monkeypatch):
        configure_intel(monkeypatch, tmp_path)

        assert not runtime.Runtime()._plan_applies()

    def test_the_pool_is_system_ram_so_two_intel_roles_coexist_by_default(self, tmp_path):
        configuration = configured(tmp_path, device="SYCL0", compute_backend="sycl")

        assert runtime.pool(configuration) == runtime.POOL_SYSTEM_RAM
        assert runtime.resolved_sharing(runtime.pool(configuration)) == runtime.SHARE_COEXIST

    def test_the_host_reserve_still_protects_the_machine(self, placed, tmp_path, monkeypatch):
        """Even with the OS limit far above it: 3 GB free and a 2 GB floor is
        1 GB safe, and a 64 MB model with an 8K cache does not fit in it."""
        configuration = configure_intel(monkeypatch, tmp_path, size_mb=64)
        budget_of(monkeypatch, host_gb=2.05)

        negotiated = runtime.negotiate(configuration)

        assert not negotiated.fits
        assert "host-RAM reserve" in negotiated.notes[-1]

    def test_warm_image_ram_is_asked_through_the_existing_admission(self, placed, server,
                                                                     tmp_path, monkeypatch):
        """The broker's host-RAM policy is the one that runs; nothing Intel-specific
        decides who gives ground."""
        managed, _started = server
        configuration = configure_intel(monkeypatch, tmp_path)
        budget_of(monkeypatch, 40)
        asked = []
        monkeypatch.setattr(mc_broker, "host_ram_fits",
                            lambda needed, reserve=None: asked.append(needed) or True)

        managed.client()

        assert asked == [configuration.model.stat().st_size]

    def test_a_vram_shortage_never_stops_an_intel_server(self, placed, server, tmp_path,
                                                         monkeypatch):
        """The broker treats a machine with one NVIDIA card as single-card and
        lets an unfiltered reclaim through; an Intel server beside that card
        holds no VRAM to give and must not be the thing that pays."""
        managed, _started = server
        configure_intel(monkeypatch, tmp_path)
        budget_of(monkeypatch, 40)
        managed.client()

        assert managed.release(1 * _GB, "an image pass", card=mc_broker.ANY_CARD) == 0
        assert managed.running()

    def test_a_restart_counts_the_memory_the_running_server_gives_back(
            self, placed, server, tmp_path, monkeypatch):
        """A vision projector arriving, or a setting change, replaces the
        running server. The budget it is admitted against excludes the RAM that
        server holds, which the replacement gets back first."""
        managed, started = server
        configuration = configure_intel(monkeypatch, tmp_path)
        weighing(monkeypatch, 17)
        budget_of(monkeypatch, host_gb=40)
        managed.client()
        assert len(started) == 1
        # The machine now reads 17 GB tighter, because the server holds it --
        # and the server's own claim on host RAM is the header's figure, as it
        # is in life, where the file on disk and its header agree.
        budget_of(monkeypatch, host_gb=6)
        monkeypatch.setattr(runtime, "host_ram_demand",
                            lambda configuration, placement=None: 17 * _GB)
        import dataclasses

        changed = dataclasses.replace(configuration, context_size=4096)
        monkeypatch.setattr(runtime, "config", lambda role="": changed)

        managed.client()

        assert len(started) == 2
        assert managed.placement().context == 4096

    def test_an_idle_intel_server_gives_its_ram_back_when_the_image_side_asks(
            self, placed, server, tmp_path, monkeypatch):
        managed, _started = server
        configure_intel(monkeypatch, tmp_path)
        budget_of(monkeypatch, 40)
        managed.client()
        size = managed.host_ram_bytes()

        freed = managed.release_host_ram(1, "an image model being read back")

        assert freed == size
        assert not managed.running()


class TestThePlacementSpeaksTheMemorysLanguage:
    def test_describe_names_the_intel_gpu_and_the_shared_memory(self):
        placement = ctx.Placement(gpu_layers=ctx.ALL_LAYERS, on_gpu=True, uma=True)

        assert placement.describe() == "all layers on the Intel GPU, in shared system memory"
        assert ctx.Placement(gpu_layers=ctx.ALL_LAYERS).describe() == "all layers on the GPU"

    def test_a_measured_speed_is_filed_apart_from_a_cuda_one(self):
        assert ctx.Placement(uma=True).speed_token == "uma"
        assert ctx.Placement().speed_token == "gpu"
        assert ctx.Placement(uma=True).key != ctx.Placement().key
        assert "shared system memory" in runtime.describe_placement_token("uma")

    def test_the_load_report_reads_sycl_buffers_as_the_device_s(self):
        offload = runtime.read_offload(
            "load_tensors: offloaded 33/33 layers to GPU\n"
            "load_tensors:        SYCL0 model buffer size = 15600.00 MiB\n"
            "load_tensors:    SYCL_Host model buffer size =   300.00 MiB\n")

        assert round(offload.device_bytes / _GB, 1) == 15.2
        assert not offload.spilled

    def test_the_report_is_never_read_as_vram_for_unified_memory(self):
        placement = ctx.Placement(gpu_layers=ctx.ALL_LAYERS, on_gpu=True, uma=True)
        offload = runtime.read_offload(
            "load_tensors:        SYCL0 model buffer size = 15600.00 MiB\n")

        assert runtime._reconciled_residency(0, 15 * _GB, offload, placement) == 0


# --------------------------------------------------------------------------- #
# 15.6 Backward compatibility
# --------------------------------------------------------------------------- #


class TestOldStateFilesResolveAsTheyAlwaysDid:
    @pytest.mark.parametrize("state, expected", [
        ({"gpu_device": "CUDA0", "mode": "gpu"}, "cuda"),
        ({"gpu_device": "CUDA1", "mode": "mixed_aggressive"}, "cuda"),
        ({"gpu_device": "CUDA0", "mode": "mixed"}, "cuda"),
        ({"gpu_device": "none", "mode": "cpu"}, "cpu"),
        ({"gpu_device": "none"}, "cpu"),
        ({"gpu_device": "CUDA0", "mode": "cpu"}, "cpu"),
        ({"gpu_device": "SYCL0"}, "sycl"),
        ({"gpu_device": "CUDA0", "runtime_id": "llama-runtime-sycl"}, "sycl"),
        ({"compute_backend": "sycl", "gpu_device": "CUDA0"}, "sycl"),
        ({"compute_backend": "cuda", "gpu_device": "SYCL0"}, "cuda"),
        ({}, "cuda"),
    ])
    def test_the_resolution_order(self, state, expected):
        assert sycl.backend_of_state(state) == expected

    def test_a_cuda_configuration_is_still_cuda(self, tmp_path):
        found = configured(tmp_path)

        assert found.backend == "cuda"
        assert found.uses_cuda_compute and not found.uses_sycl_compute
        assert runtime.card_of(found) == 0

    def test_a_cpu_configuration_is_still_the_processor(self, tmp_path):
        found = configured(tmp_path, device="none", mode="cpu")

        assert found.backend == "cpu"
        assert not found.uses_cuda_compute and not found.uses_sycl_compute
        assert runtime.card_of(found) is None

    def test_an_intel_configuration_names_no_cuda_card(self, tmp_path):
        found = configured(tmp_path, device="SYCL0", compute_backend="sycl")

        assert found.uses_sycl_compute and not found.uses_cuda_compute
        assert found.on_gpu
        assert runtime.card_of(found) is None

    def test_the_3090_and_5090_keep_their_families(self):
        ampere = GpuInfo(0, "GPU-0", "NVIDIA GeForce RTX 3090", 24576, 23304, "560", 8.6)
        blackwell = GpuInfo(1, "GPU-1", "NVIDIA GeForce RTX 5090", 32768, 32000, "570", 12.0)

        assert pinned.runtime_component_id(ampere) == "llama-runtime-cuda12"
        assert pinned.runtime_component_id(blackwell) == "llama-runtime-cuda13"
        assert pinned.runtime_component_ids(ampere) == ("llama-runtime-cuda12",
                                                        "llama-runtime-cuda12-cudart")

    def test_a_cuda_start_still_reads_its_card_and_measures_its_vram(self, placed, server,
                                                                     tmp_path, monkeypatch):
        """The Intel path took the card reading away from the placements that
        have no card; the one that has must keep it."""
        managed, _started = server
        model = build_model(tmp_path, blocks=32, size_mb=64)
        binary = tmp_path / "llama-server"
        binary.write_bytes(b"")
        configuration = runtime.Config(
            runtime=binary, model=model, mmproj=None, gpu_index=0, device="CUDA0",
            gpu_layers="all", context_size=8192, context_mode="fixed", context_buffer_gb=4.0,
            kv_type_k="f16", kv_type_v="f16", mode="gpu")
        monkeypatch.setattr(runtime, "config", lambda role="": configuration)
        monkeypatch.setattr(mc_broker, "free_vram_bytes", lambda: 20 * _GB)
        card = {"free": 20 * _GB}
        monkeypatch.setattr(mc_broker, "device_free_vram_bytes", lambda index=None: card["free"])
        started: list = []

        def taking():
            process = FakeProcess(started)
            begin = process.start

            def start_and_take(*args, **kwargs):
                begin(*args, **kwargs)
                card["free"] = 19 * _GB  # the server took a gigabyte

            process.start = start_and_take
            return process

        monkeypatch.setattr(managed, "_new_process", taking)

        managed.client()

        assert managed.resident_bytes() == 1 * _GB
        assert not managed.placement().uma

    def test_a_processor_start_no_longer_prints_a_card_s_free_vram(self, placed, server,
                                                                   tmp_path, monkeypatch,
                                                                   caplog):
        """From a user's log: "system RAM (no GPU offload) ... 5.8 GB VRAM" for a
        server holding none, read off the image card while a checkpoint loaded."""
        managed, _started = server
        model = build_model(tmp_path, blocks=32, size_mb=64)
        binary = tmp_path / "llama-server"
        binary.write_bytes(b"")
        configuration = runtime.Config(
            runtime=binary, model=model, mmproj=None, gpu_index=-1, device="none",
            gpu_layers="0", context_size=8192, context_mode="fixed", context_buffer_gb=4.0,
            kv_type_k="f16", kv_type_v="f16", mode="cpu")
        monkeypatch.setattr(runtime, "config", lambda role="": configuration)
        readings = iter([20 * _GB])
        monkeypatch.setattr(mc_broker, "device_free_vram_bytes",
                            lambda index=None: next(readings, 14 * _GB))
        monkeypatch.setattr(mc_broker, "free_ram_bytes", lambda: 60 * _GB)

        with caplog.at_level("INFO", logger="model_chain"):
            managed.client()

        start = next(r.getMessage() for r in caplog.records if "starting llama-server" in
                     r.getMessage())
        ready = next(r.getMessage() for r in caplog.records if "llama-server ready" in
                     r.getMessage())
        assert "60.0 GB of system RAM free" in start
        assert "0.0 GB VRAM" in ready
        assert managed.resident_bytes() == 0
